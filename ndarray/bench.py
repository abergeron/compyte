import gc, time, math, re
import numpy
import numexpr
import theano
theano.config.floatX = 'float32'

try:
#    import theano.misc.pycuda_init# To allow selecting the gpu with THEANO_FLAGS
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    from pycuda.elementwise import ElementwiseKernel
    from pycuda import gpuarray
except ImportError:
    pass

try:
    import gen_elemwise
    from gen_elemwise import ElemwiseAlgo, call_elemwise, MyGpuNdArray
    import pygpu_ndarray as gpu_ndarray
except ImportError:
    pass

def timeit_core(f, loops, nogc):
    gc.collect()
    if nogc:
        gc.disable()
    try:
        tstart = time.time()
        for _ in xrange(loops):
            f()
        tend = time.time()
    finally:
        if nogc:
            gc.enable()
        gc.collect()

    return (tend - tstart)/loops

def timeit(f, lbl, n_tries=3, nogc=True):
    gc.disable()
    try:
        ts = time.time()
        f()
        te = time.time()
    finally:
        gc.enable()
        gc.collect()
    est = te - ts

    # This will generate a runtime between 1 and 10 seconds
    loops = max(1, int(10**math.floor(math.log(10/est, 10))))
    
    mintime = min(timeit_core(f, loops, nogc) for _ in xrange(n_tries))
    
    print lbl, "(", loops, "loops ):", mintime, "s ( best of", n_tries, ")"
    return mintime

numpy_tmpl = r"""
def numpy(%s):
    return %s
"""
numpy_map = {
    'sin': numpy.sin
}

numexpr_tmpl = r"""
def numexpr(vars):
    return evaluate("%s", local_dict=vars)
"""
numexpr_map = {
    'evaluate': numexpr.evaluate
    }

theano_tmpl = r"""
exp = %s
"""
theano_smap = {
    'sin': theano.scalar.sin
}

theano_tmap = {
    'sin': theano.tensor.sin
}

theano_support = r"""
typedef float npy_float32;
typedef double npy_float64;
"""

def empty(vars):
    return None

def elemwise_helper(kern, vars):
    import theano
    s = dict((k, theano.scalar.ScalarVariable(theano.scalar.Scalar(str(v.dtype)))) for k, v in vars.iteritems())

    theano_code = theano_tmpl%(kern.exp,)
    res = s.copy()
    exec theano_code in theano_smap, res
    sexp = res['exp']

    order = vars.keys()
    inputs_s = [s[k] for k in order]
    comp = theano.scalar.Composite(inputs_s, [sexp])
    
    fct = MyGpuNdArray.gen_fct(theano.tensor.Elemwise(comp),
                               [MyGpuNdArray(i) for i in vars.values()],
                               vars.values()[0].ndim)

    def ndarray(gvars, out=None):
        inputs = [gvars[k] for k in order]
        return fct(inputs, out=out)

    return ndarray

def var_iter(vars):
    """ return the initial value for the inputs.
    
    This is done this way to ensure we keep only the mininum memory footprint.
    """
    keys_order = vars.keys()
    for shape_idx in range(len(vars.items()[0][1])):
        ret = {}
        for key in vars.keys():
            shape = [v[0] if isinstance(v, (list, tuple)) else v for v in vars[key][shape_idx]]
            stride = [v[1] if isinstance(v, (list, tuple)) else 1 for v in vars[key][shape_idx]]

            val = numpy.random.random(shape).astype('float32')
            ret[key] = val, stride
        yield ret

def pycuda_helper(kern, vars):
    # this does float32 only
    order = vars.keys()
    kern = re.sub(r'([0-9a-z.]+)\*\*([0-9a-z.]+)', r'pow(\1, \2)', kern)
    for k in order:
        kern = kern.replace(k, k+'[i]')
    kern = '__out[i] = '+kern
    ek = ElementwiseKernel('float *__out,' + ','.join("const float *%s"%(n,) for n in order), kern)
    def pycuda(vars, out=None):
        inputs = [vars[k] for k in order]
        if out is None or out.dtype != inputs[0].dtype or out.shape != inputs[0].shape:
            out = gpuarray.empty_like(inputs[0])
        ek(out, *inputs)
        return out
    return pycuda

def apply_strides(vars):
    vars2 = {}
    for k, (val, stride) in vars.iteritems():
        vars2[k] = val[tuple(slice(None, None, st) for st in stride)]
    return vars2

class bench(object):
    def __init__(self, exp, **vars):
        self.exp = exp
        self.vars = vars
        self.empty = empty
        numpy_code = numpy_tmpl%(','.join(vars.keys()), exp)
        res = {}
        exec numpy_code in numpy_map, res
        self.numpy = res['numpy']
        numexpr_code = numexpr_tmpl%(exp,)
        res = {}
        exec numexpr_code in numexpr_map, res
        self.numexpr = res['numexpr']
        self.pycuda = pycuda_helper(self.exp, self.vars)

    def run(self):
        res = []
        for vals in var_iter(self.vars):
            print [(k, v.shape) for k, v, in vals.items()]
            res.append(self.one_try(vals))
        return res

    def try_numpy(self, vars, retval=None):
        vars = apply_strides(vars)
        t = timeit(lambda: self.numpy(**vars), "numpy   ")
        if retval is not None:
            retval['numpy'] = t
        return t
    
    def try_numexpr(self, vars, ref=None, retval=None):
        # to cache the expression, since we don't want to time the compile
        try:
            vars = apply_strides(vars)
            if ref is not None:
                res = self.numexpr(vars)
                assert (res == ref).all()
            t = timeit(lambda: self.numexpr(vars), "numexpr ")
            if retval is not None:
                retval['numexpr'] = t
            return t
        except Exception, e:
            print "numexpr: error", e
            import traceback
            traceback.print_exc()

    def try_compyte(self, vars, ref=None, retval=None, reuse_output=False):
        try:
            # Take care! gpu_ndarray.GpuNdArrayObject will always generate a c contiguous
            # memory region! This bypass the strides tests!
            gvars = dict((k, (gpu_ndarray.GpuNdArrayObject(v), st))
                         for k,(v,st) in vars.iteritems())
            gvars = apply_strides(gvars)
            self.ndarray = elemwise_helper(self, gvars)
            res = self.ndarray(gvars)
            print 'Dimensions after collapse', gen_elemwise.elemwise_collapses(gvars.values(),[res])[0]
            if ref is not None:
                assert numpy.allclose(res, ref)
            if reuse_output:
                out = res
            else:
                out = None
            t = timeit(lambda: self.ndarray(gvars, out), "compyte ", nogc=False)
            if retval is not None:
                retval['compyte'] = t
            return t
        except Exception, e:
            print "compyte: error", e
            import traceback
            traceback.print_exc()

    def try_pycuda(self, vars, ref=None, retval=None, reuse_output=False):
        try:
            assert (numpy.asarray([st for k,(v,st) in vars.iteritems()])==1).all()
            pvars = dict((k, gpuarray.to_gpu(v)) for k,(v,st) in vars.iteritems())
            res = self.pycuda(pvars)
            if ref is not None:
                assert numpy.allclose(res.get(), ref)
            if reuse_output:
                out = res
            else:
                out = None
            t = timeit(lambda: self.pycuda(pvars, out), "pycuda  ", nogc=False)
            if retval is not None:
                retval['pycuda'] = t
            return t
        except Exception, e:
            print "pycuda: error", e
            import traceback
            traceback.print_exc()
    
    def one_try(self, vars):
        retval = {}
        timeit(lambda: self.empty(vars), "baseline")

        ref = self.numpy(**vars)
        retval['numpy'] = timeit(lambda: self.numpy(**vars), "numpy   ")

        self.try_numexpr(vars, ref=ref, retval=retval)
        self.try_compyte(vars, ref=ref, retval=retval)
        self.try_pycuda(vars, ref=ref, retval=retval)

        return retval

def prod(seq):
    s = 1
    for x in seq:
        s *= x
    return s

MARKERS = ['+', '*', ',', '.', '1', '2', '3', '4', '<', '>', 'D', 'H', '^', '_', 'd', 'h', 'o', 'p', 's', 'v', 'x', '|']
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

def make_graph(name, b, msa, times={}):
    ext = 'pdf'
    import matplotlib
    matplotlib.use(ext, warn=False) # maybe Agg or Cairo
    import matplotlib.pyplot as plt
    print 'Start graph', name
    idx = 0
    for lbl, m, shapes in msa:
        vars = {}
        for k in b.vars.keys():
            vars[k] = shapes
        xvals = []
        yvals = []
        print lbl
        if lbl in times:
            assert len(times[lbl]) == len(shapes)
            yvals = numpy.asarray(times[lbl])*1e6
        for vals_strides in var_iter(vars):
            xvals.append(prod(vals_strides.values()[0][0].shape))
            if lbl not in times:
                #ref = b.numpy(**vals)
                ref = None
                yvals.append(m(b, vals_strides, ref=ref)*1e6)
        plt.semilogx(xvals, yvals, label=lbl,
                     color=COLORS[idx], marker=MARKERS[idx])
        idx += 1
    plt.legend(loc='upper left')
    plt.ylabel('time (us)')
    plt.xlabel('number of elements')
    plt.title(b.exp)
    filename = name+'.'+ext
    plt.savefig(filename)
    print "saved in file", filename
    plt.cla()
    plt.clf()

#Too much mem on oolong: (1e6,), (1e8)))
#shapes = ((100,), (100000,), (100,1000), (100,100,10),(100,10,10,10))
#shapes = ((100,10,10,10),)
shapes = ((1e2,), (1e3,))#, (1e4,), (1e5,), (1e6,), (1e7,))
ap1 = bench("a+1", a=shapes)
apb = bench("a+b", a=shapes, b=shapes)
b2 = bench("2*a + 3*b", a=((100,),), b=((100,),))
b3 = bench("a**2 + b**2 + 2*a*b", a=((100,),), b=((100,),))
b4 = bench("2*a + b**10", a=((100,),), b=((100,),))
series = bench("2*sin(a)-sin(2*a)+2/3.0*sin(3*a)-1/2.0*sin(4*a)+2/5.0*sin(5*a)-1/3.0*sin(6*a)+2/7.0*sin(7*a)",
               a=((100,), (1000,), (100, 200)))

if __name__ == "__main__":
    msa = [('pycuda', bench.try_pycuda, ((100,), (1000,), (10000,), (100000,), (1000000,), (10000000,))),
           ('numpy 1d', bench.try_numpy, ((100,), (1000,), (10000,), (100000,), (1000000,))),#, (10000000,))),
           ('compyte 1d', bench.try_compyte, ((100,), (1000,), (10000,), (100000,), (1000000,), (10000000,))),
           #('compyte 2d', bench.try_compyte, ((10, 10), (100, 10), (100, 100), (1000, 100), (1000, 1000), (10000, 1000)), None, None, True),
           #('compyte 3d', bench.try_compyte, ((10, 10, 1), (10, 10, 10), (100, 10, 10), (100, 100, 10), (100, 100, 100), (1000, 100, 100)), None, None, True),
           ('compyte 4d', bench.try_compyte, ((10, 10, 1, 1), (10, 10, 10, 1), (10, 10, 10, 10), (100, 10, 10, 10), (100, 100, 10, 10), (100, 100, 100, 10)))]
    msa2 = [('pycuda', lambda b, vals, ref: b.try_pycuda(vals, reuse_output=True, ref=ref), ((100,), (1000,), (10000,), (100000,), (1000000,), (10000000,))),
            ('compyte 1d',  lambda b, vals, ref: b.try_compyte(vals, reuse_output=True, ref=ref),  ((100,), (1000,), (10000,), (100000,), (1000000,), (10000000,))),
            ('compyte 4d', lambda b, vals, ref: b.try_compyte(vals, reuse_output=True, ref=ref), ((10, 10, 1, 1), (10, 10, 10, 1), (10, 10, 10, 10), (100, 10, 10, 10), (100, 100, 10, 10), (100, 100, 100, 10))),
            ('compyte 4d strided', lambda b, vals, ref: b.try_compyte(vals, reuse_output=True, ref=ref), (((20, 2), (20, 2), (2, 2), (2, 2)), ((20, 2), (20, 2), (20, 2), (2, 2)), ((20, 2), (20, 2), (20, 2), (20, 2)), ((200, 2), (20, 2), (20, 2), (20, 2)), ((200, 2), (200, 2), (20, 2), (20, 2)), ((200, 2), (200, 2), (200, 2), (20, 2))))]

    for suffix, m in [('', msa),('_no_alloc', msa2)]:
        make_graph('ap1'+suffix, ap1, msa=m)
        make_graph('apb'+suffix, apb, msa=m)
        make_graph('2ap3b'+suffix, b2, msa=m)
        make_graph('a2pb2p2ab'+suffix, b3, msa=m)
        make_graph('2apb10'+suffix, b4, msa=m)
        make_graph('series'+suffix, series, msa=m)
