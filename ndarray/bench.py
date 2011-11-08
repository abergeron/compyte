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

def timeit_core(f, loops, nogc, sync_gpu):
    gc.collect()
    if nogc:
        gc.disable()
    try:
        if sync_gpu:
            pycuda._driver.Context.synchronize()
        tstart = time.time()
        for _ in xrange(loops):
            f()
        if sync_gpu:
            pycuda._driver.Context.synchronize()
        tend = time.time()
    finally:
        if nogc:
            gc.enable()
        gc.collect()

    return (tend - tstart)/loops

def timeit(f, lbl, n_tries=3, nogc=True, sync_gpu=False):
    # sync_gpu sync before/after for the first estimation call
    # and before/after the loops.
    # The after sync time overhead is included in the timming.
    gc.disable()
    try:
        if sync_gpu:
            pycuda._driver.Context.synchronize()
            ts = time.time()
            f()
            pycuda._driver.Context.synchronize()
            te = time.time()
        else:
            ts = time.time()
            f()
            te = time.time()
    finally:
        gc.enable()
        gc.collect()
    est = te - ts

    # This will generate a runtime between 1 and 10 seconds
    loops = max(1, int(10**math.floor(math.log(10/est, 10))))
    
    times = [timeit_core(f, loops, nogc, sync_gpu) for _ in xrange(n_tries)]
    mintime = min(times)
    total_time = (numpy.asarray(times)*loops).sum()
    print lbl, "(%d loops): %.3es (best of %d) run time %.2f"%(loops, mintime, n_tries, total_time)
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

def elemwise_helper(kern, vars, collapse=True):
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
                               vars.values()[0].ndim,
                               collapse=collapse)

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

    def try_compyte(self, vars, ref=None, retval=None, reuse_output=False, collapse=True):
        try:
            # Take care! gpu_ndarray.GpuNdArrayObject will always generate a c contiguous
            # memory region! This bypass the strides tests!
            gvars = dict((k, (gpu_ndarray.GpuNdArrayObject(v), st))
                         for k,(v,st) in vars.iteritems())
            gvars = apply_strides(gvars)
            self.ndarray = elemwise_helper(self, gvars, collapse=collapse)
            res = self.ndarray(gvars)
            nd_col, info = gen_elemwise.elemwise_collapses(gvars.values(),[res])
            print 'Size/Dimensions/Shapes after collapse:', numpy.prod(res.shape), nd_col, info[0][:nd_col]#, numpy.asarray(info[1])[:,:nd_col]
            if ref is not None:
                assert numpy.allclose(res, ref)
            if reuse_output:
                out = res
            else:
                out = None
            t = timeit(lambda: self.ndarray(gvars, out), "compyte ", nogc=False, sync_gpu=True)
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
            t = timeit(lambda: self.pycuda(pvars, out), "pycuda  ", nogc=False, sync_gpu=True)
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
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'grey', 'yellow', 'k', 'orange', 'amber', 'purple', 'pink', '#FFAABB', '#FFAAFF']
##FFAABB, #FFAAFF
def make_graph(name, b, msa, times={}):
    ext = 'pdf'
    import matplotlib
    matplotlib.use(ext, warn=False) # maybe Agg or Cairo
    import matplotlib.pyplot as plt
    print 'Start graph', name
    idx = 0
    for lbl, m, shapes in msa:
        print
        vars = {}
        for k in b.vars.keys():
            vars[k] = shapes
        xvals = []
        yvals = []
        print lbl
        if lbl in times:
            assert len(times[lbl]) == len(shapes)
            yvals = numpy.asarray(times[lbl])
        for vals_strides in var_iter(vars):
            xvals.append(prod(numpy.asarray(vals_strides.values()[0][0].shape)/
                              vals_strides.values()[0][1]))
            if lbl not in times:
                #ref = b.numpy(**vals)
                ref = None
                yvals.append(m(b, vals_strides, ref=ref))
        print "nb elem", xvals
        print "times", yvals
        yvals = [y*1e6 for y in yvals]
        assert len(xvals) == len(yvals)
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


def gen_submited_paper_pdf():
    ap1_no_alloc_submit_biglearn_paper = {
        'pycuda': [2.263465166091919e-05, 2.2561240196228026e-05, 5.0088729858398435e-05, 0.00024293808937072753, 0.00048887078762054439],
        'compyte 1d contiguous': [4.8440003395080566e-05, 4.8550620079040524e-05, 5.4675912857055667e-05, 0.00027481579780578615, 0.00056338310241699219],
        'compyte 4d contiguous': [6.9326648712158203e-05, 6.9620189666748047e-05, 7.0056605339050297e-05, 0.00027470300197601316, 0.00056341369152069088],
        'compyte 4d contiguous not collapsed': [9.3558812141418462e-05, 9.4482517242431646e-05, 0.00029041900634765627, 0.00144551682472229, 0.002890136957168579],
        'compyte 4d strided outer(2d after collapse)': [8.9702391624450685e-05, 9.054498672485351e-05, 0.00010384140014648437, 0.00051974868774414061, 0.00092990517616271969],
        }

    a2pb2p2ab_no_alloc_no_alloc_submit_biglearn_paper = {
        'pycuda': [2.3609220981597899e-05, 2.3830621242523194e-05, 9.885969161987305e-05, 0.00047158560752868654, 0.00093993830680847172],
        'compyte 1d contiguous': [5.5107808113098146e-05, 5.5577740669250487e-05, 0.00011780040264129639, 0.00053009800910949702, 0.0010585570335388183],
        'compyte 4d contiguous': [8.4473395347595213e-05, 8.4439301490783696e-05, 0.00011779539585113526, 0.00053011291027069086, 0.0010585739612579345],
        'compyte 4d contiguous not collapsed': [0.00012182121276855469, 0.0001221674919128418, 0.00034283380508422852, 0.0016802849769592286, 0.0033540680408477783],
        'compyte 4d strided outer(2d after collapse)': [0.00011260590553283692, 0.00011380398273468018, 0.00021288449764251708, 0.00092618393898010257, 0.0018587779998779296],
        }

    shapes_1d = [(100,), (1000,), (10000,), (100000,), (1000000,), (5000000,), (10000000,), (50000000,)]
    shapes_2d = [(10, 10), (100, 10), (100, 100), (1000, 100), (1000, 1000),
                 (5000, 1000), (10000, 1000), (50000, 1000)]
    shapes_3d = [(10, 10, 1), (10, 10, 10), (100, 10, 10), (100, 100, 10),
                 (100, 100, 100), (500, 100, 100), (1000, 100, 100),
                 (5000, 100, 100)]
    shapes_4d = [(10, 10, 1, 1), (10, 10, 10, 1), (10, 10, 10, 10),
                 (100, 10, 10, 10), (100, 100, 10, 10), (500, 100, 10, 10),
                 (100, 100, 100, 10), (500, 100, 100, 10)]
    shapes_4d_strided_outer_coalesced = [((8, 2), 1, 1, 32),
                                         ((8, 2), 10, 1, 32),
                                         ((8, 2), 10, 10, 32),
                                         ((62, 2), 10, 10, 32),
                                         ((62, 2), 100, 10, 32),
                                         ((312, 2), 100, 10, 32),
                                         ((62, 2), 100, 100, 32),
                                         ((312, 2), 100, 100, 32)]
    shapes_4d_strided_outer = [((20, 2), 10, 1, 1),
                               ((20, 2), 10, 10, 1),
                               ((20, 2), 10, 10, 10),
                               ((200, 2), 10, 10, 10),
                               ((200, 2), 100, 10, 10),
                               ((1000, 2), 100, 10, 10),
                               ((200, 2), 100, 100, 10),
                               ((1000, 2), 100, 100, 10)]
    shapes_4d_strided_inner = [(10, 10, 1, (20, 2)),
                               (10, 10, 10, (20, 2)),
                               (10, 10, 10, (20, 2)),
                               (10, 10, 10, (200, 2)),
                               (10, 100, 10, (200, 2)),
                               (10, 100, 10, (1000, 2)),
                               (10, 100, 100, (200, 2)),
                               (10, 100, 100, (1000, 2))]
    shapes_4d_strided_middle = [(10, (20, 2), 10, 1),
                               (10, (20, 2), 10, 10),
                               (10, (20, 2), 10, 10),
                               (10, (200, 2), 10, 10),
                               (10, (200, 2), 100, 10),
                               (10, (1000, 2), 100, 10),
                               (10, (200, 2), 100, 100),
                               (10, (1000, 2), 100, 100)]
    shapes_4d_strided_middle2 = [(10, 10, 1, (20, 2), 1),
                               (10, 10, (20, 2), 10),
                               (10, 10, (20, 2), 10),
                               (10, 10, (200, 2), 10),
                               (10, 100, (200, 2), 10),
                               (10, 100, (1000, 2), 10),
                               (10, 100, (200, 2), 100),
                               (10, 100, (1000, 2), 100)]
    shapes_4d_strided4d = [((20, 2), (20, 2), (2, 2), (2, 2)),
                           ((20, 2), (20, 2), (20, 2), (2, 2)),
                           ((20, 2), (20, 2), (20, 2), (20, 2)),
                           ((200, 2), (20, 2), (20, 2), (20, 2)),
                           ((200, 2), (200, 2), (20, 2), (20, 2)),
                           ((1000, 2), (200, 2), (20, 2), (20, 2)),
                           ((200, 2), (200, 2), (200, 2), (20, 2)),
                           ]
                           #((1000, 2), (200, 2), (200, 2), (20, 2))] # 3G per object

    # Remove the case 100 et 1000 elements
    all_shapes = [shapes_1d,
                  shapes_2d,
                  shapes_3d,
                  shapes_4d,
                  shapes_4d_strided_outer_coalesced,
                  shapes_4d_strided_outer,
                  shapes_4d_strided_inner,
                  shapes_4d_strided_middle,
                  shapes_4d_strided_middle2,
                  shapes_4d_strided4d]
    if True:
        for x in all_shapes:
            x.remove(x[0])
            x.remove(x[0])

    # Remove the last case (50000000 elements)
    # as this hide the detail when we have few elements
    if True:
        for x in all_shapes:
            x.remove(x[-1])

    # Remove the shape that take more then 700M of inputs
    # We need a minimum of 1 inputs and 2 outputs!
    if True:
        for x in all_shapes:
            for shape in x:
                if numpy.prod([ s if not isinstance(s, (tuple,list)) else s[0] for s in shape])*4>700e6:
                    x.remove(shape)

    msa2 = [('pycuda',
             lambda b, vals, ref: b.try_pycuda(vals, reuse_output=True, ref=ref),
             shapes_1d),
            ('compyte 1d contiguous',
             lambda b, vals, ref: b.try_compyte(vals, reuse_output=True, ref=ref),
             shapes_1d),
            ('compyte 4d contiguous',
             lambda b, vals, ref: b.try_compyte(vals, reuse_output=True, ref=ref),
             shapes_4d),
            ('compyte 4d contiguous not collapsed',
             lambda b, vals, ref: b.try_compyte(vals, reuse_output=True, ref=ref, collapse=False),
             shapes_4d),
            ('compyte 4d strided outer(2d after collapse)',
             lambda b, vals, ref: b.try_compyte(vals, reuse_output=True, ref=ref),
             shapes_4d_strided_outer),
            ]

    for suffix, m in [('_no_alloc', msa2)]:
        make_graph('ap1'+suffix, ap1, msa=m, times=ap1_no_alloc_submit_biglearn_paper) # paper submited
        make_graph('a2pb2p2ab'+suffix, b3, msa=m, times=a2pb2p2ab_no_alloc_no_alloc_submit_biglearn_paper) # paper


if __name__ == "__main__":
    if False:
        gen_submited_paper_pdf()
        sys.exit()

    shapes_1d = [(100,), (1000,), (10000,), (100000,), (1000000,), (5000000,), (10000000,), (50000000,)]
    shapes_2d = [(10, 10), (100, 10), (100, 100), (1000, 100), (1000, 1000),
                 (5000, 1000), (10000, 1000), (50000, 1000)]
    shapes_3d = [(10, 10, 1), (10, 10, 10), (100, 10, 10), (100, 100, 10),
                 (100, 100, 100), (500, 100, 100), (1000, 100, 100), (5000, 100, 100)]
    shapes_4d = [(10, 10, 1, 1), (10, 10, 10, 1), (10, 10, 10, 10),
                 (100, 10, 10, 10), (100, 100, 10, 10), (500, 100, 10, 10),
                 (100, 100, 100, 10), (500, 100, 100, 10)]
    shapes_4d_strided_outer_coalesced = [((8, 2), 1, 1, 32),
                                         ((8, 2), 10, 1, 32),
                                         ((8, 2), 10, 10, 32),
                                         ((62, 2), 10, 10, 32),
                                         ((62, 2), 100, 10, 32),
                                         ((312, 2), 100, 10, 32),
                                         ((62, 2), 100, 100, 32),
                                         ((312, 2), 100, 100, 32)]
    shapes_4d_strided_outer = [((20, 2), 10, 1, 1),
                               ((20, 2), 10, 10, 1),
                               ((20, 2), 10, 10, 10),
                               ((200, 2), 10, 10, 10),
                               ((200, 2), 100, 10, 10),
                               ((1000, 2), 100, 10, 10),
                               ((200, 2), 100, 100, 10),
                               ((1000, 2), 100, 100, 10)]
    shapes_4d_strided_inner = [(10, 10, 1, (20, 2)),
                               (10, 10, 10, (20, 2)),
                               (10, 10, 10, (20, 2)),
                               (10, 10, 10, (200, 2)),
                               (10, 100, 10, (200, 2)),
                               (10, 100, 10, (1000, 2)),
                               (10, 100, 100, (200, 2)),
                               (10, 100, 100, (1000, 2))]
    shapes_4d_strided_middle = [(10, (20, 2), 10, 1),
                               (10, (20, 2), 10, 10),
                               (10, (20, 2), 10, 10),
                               (10, (200, 2), 10, 10),
                               (10, (200, 2), 100, 10),
                               (10, (1000, 2), 100, 10),
                               (10, (200, 2), 100, 100),
                               (10, (1000, 2), 100, 100)]
    shapes_4d_strided_middle2 = [(10, 10, 1, (20, 2), 1),
                               (10, 10, (20, 2), 10),
                               (10, 10, (20, 2), 10),
                               (10, 10, (200, 2), 10),
                               (10, 100, (200, 2), 10),
                               (10, 100, (1000, 2), 10),
                               (10, 100, (200, 2), 100),
                               (10, 100, (1000, 2), 100)]
    shapes_4d_strided4d = [((20, 2), (20, 2), (2, 2), (2, 2)),
                           ((20, 2), (20, 2), (20, 2), (2, 2)),
                           ((20, 2), (20, 2), (20, 2), (20, 2)),
                           ((200, 2), (20, 2), (20, 2), (20, 2)),
                           ((200, 2), (200, 2), (20, 2), (20, 2)),
                           ((1000, 2), (200, 2), (20, 2), (20, 2)),
                           ((200, 2), (200, 2), (200, 2), (20, 2)),
                           ]
                           #((1000, 2), (200, 2), (200, 2), (20, 2))] # 3G per object

    # Remove the case 100 et 1000 elements
    all_shapes = [shapes_1d,
                  shapes_2d,
                  shapes_3d,
                  shapes_4d,
                  shapes_4d_strided_outer_coalesced,
                  shapes_4d_strided_outer,
                  shapes_4d_strided_inner,
                  shapes_4d_strided_middle,
                  shapes_4d_strided_middle2,
                  shapes_4d_strided4d]
    if True:
        for x in all_shapes:
            x.remove(x[0])
            x.remove(x[0])

    # Remove the last case (50000000 elements)
    # as this hide the detail when we have few elements
    if True:
        for x in all_shapes:
            x.remove(x[-1])

    # Remove the shape that take more then 700M of inputs
    # We need a minimum of 1 inputs and 2 outputs!
    if True:
        for x in all_shapes:
            for shape in x:
                if numpy.prod([ s if not isinstance(s, (tuple,list)) else s[0] for s in shape])*4>700e6:
                    x.remove(shape)

    times_ap1 = {'pycuda':[2.19774413109e-05, 2.19221115112e-05, 4.95604705811e-05, 0.000240413618088, 0.000483848741055, 0.0024825881815],
             'compyte 1d contiguous':[5.62311887741e-05, 5.63329601288e-05, 5.65941119194e-05, 0.000271942930222, 0.000558048298359, 0.00291627338171],
             'compyte 4d contiguous':[6.96398019791e-05, 6.97722792625e-05, 6.97479104996e-05, 0.000272047069073, 0.000557991290092, 0.00264423480034],
             #'compyte 4d strided outer coallesced':[],
             #'compyte 4d strided outer':[],
             #'compyte 4d strided 4d':[],
                 }

    msa = [('pycuda', bench.try_pycuda, shapes_1d),
#           ('numpy 1d', bench.try_numpy, shapes_1d[:-1]),
           ('compyte 1d contiguous', bench.try_compyte, shapes_1d),
           #('compyte 2d', bench.try_compyte, shapes_2d),
           #('compyte 3d', bench.try_compyte, shapes_3d),
           ('compyte 4d contiguous', bench.try_compyte, shapes_4d)]
    msa2 = [('pycuda',
             lambda b, vals, ref: b.try_pycuda(vals, reuse_output=True, ref=ref),
             shapes_1d),
            ('compyte 1d contiguous',
             lambda b, vals, ref: b.try_compyte(vals, reuse_output=True, ref=ref),
             shapes_1d),
#            ('compyte 1d contiguous not collapsed',
#             lambda b, vals, ref: b.try_compyte(vals, reuse_output=True, ref=ref, collapse=False),
#             shapes_1d),
            ('compyte 4d contiguous',
             lambda b, vals, ref: b.try_compyte(vals, reuse_output=True, ref=ref),
             shapes_4d),
            ('compyte 4d contiguous not collapsed',
             lambda b, vals, ref: b.try_compyte(vals, reuse_output=True, ref=ref, collapse=False),
             shapes_4d),
#            ('compyte 4d strided outer coallesced',
#             lambda b, vals, ref: b.try_compyte(vals, reuse_output=True, ref=ref),
#             shapes_4d_strided_outer_coalesced),
#            ('compyte 4d strided middle(2d after collapse)',
#             lambda b, vals, ref: b.try_compyte(vals, reuse_output=True, ref=ref),
#             shapes_4d_strided_middle),
#            ('compyte 4d strided middle2(2d after collapse)',
#             lambda b, vals, ref: b.try_compyte(vals, reuse_output=True, ref=ref),
#             shapes_4d_strided_middle),
            ('compyte 4d strided outer(2d after collapse)',
             lambda b, vals, ref: b.try_compyte(vals, reuse_output=True, ref=ref),
             shapes_4d_strided_outer),
#            ('compyte 4d strided inner(1d after collapse)',
#             lambda b, vals, ref: b.try_compyte(vals, reuse_output=True, ref=ref),
#             shapes_4d_strided_inner),
            #('compyte 4d strided 4d',
            # lambda b, vals, ref: b.try_compyte(vals, reuse_output=True, ref=ref),
            # shapes_4d_strided4d),
            ]

    for suffix, m in [('_no_alloc', msa2)]:#[('', msa),('_no_alloc', msa2)]:
        make_graph('ap1'+suffix, ap1, msa=m)#, times=times_ap1) # paper
        make_graph('a2pb2p2ab'+suffix, b3, msa=m) # paper
#        make_graph('series'+suffix, series, msa=m)
#        make_graph('apb'+suffix, apb, msa=m)
#        make_graph('2ap3b'+suffix, b2, msa=m)
#        make_graph('2apb10'+suffix, b4, msa=m)
