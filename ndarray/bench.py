import gc, time, math
import numpy
import numexpr
import theano
theano.config.floatX = 'float32'

try:
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

def timeit_core(f, loops):
# pycuda can't handle no gc beyond a certain number of calls
#    gc.disable()
    tstart = time.time()
    for _ in xrange(loops):
        f()
    tend = time.time()
#    gc.enable()
    gc.collect()

    return (tend - tstart)/loops

def timeit(f, lbl, n_tries=3):
    gc.disable()
    ts = time.time()
    f()
    te = time.time()
    gc.enable()
    gc.collect()
    est = te - ts

    loops = max(1, int(10**math.floor(math.log(10/est, 10))))
    
    mintime = min(timeit_core(f, loops) for _ in xrange(n_tries))
    
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
    t = dict((k, theano.tensor.TensorVariable(theano.tensor.TensorType(str(v.dtype), (False,)*len(v.shape)))) for k, v in vars.iteritems())

    theano_code = theano_tmpl%(kern.exp,)
    res = s.copy()
    exec theano_code in theano_smap, res
    sexp = res['exp']
    res = t.copy()
    exec theano_code in theano_tmap, res
    texp = res['exp']
    order = vars.keys()
    inputs_s = [s[k] for k in order]
    comp = theano.scalar.Composite(inputs_s, [sexp])
    alg = ElemwiseAlgo(comp)
    
    inputs_t = [t[k] for k in order]
    nodename = 'bench'
    klen = len(vars.values()[0].shape)
    mod = SourceModule(theano_support + alg.c_src_kernel(inputs_t, [texp], nodename, klen, static=""))
    fct = mod.get_function("kernel_%s_%d"%(nodename, klen))

    def ndarray(gvars):
        inputs = [gvars[k] for k in order]
#        gen_elemwise.pycuda_array = False
        return call_elemwise(fct, inputs, block=(inputs[0].shape[-1], 1, 1))
    return ndarray

def pycuda_helper(kern, vars):
    # this does float32 only
    order = vars.keys()
    for k in order:
        kern = kern.replace(k, k+'[i]')
    kern = '__out[i] = '+kern
    ek = ElementwiseKernel('float *__out,' + ','.join("float *%s"%(n,) for n in order), kern)
    def pycuda(vars):
        inputs = [vars[k] for k in order]
        out = gpuarray.empty_like(inputs[0])
        ek(out, *inputs)
        return out
    return pycuda

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
        keys = self.vars.keys()
        for shapes in zip(*[self.vars[k] for k in keys]):
            vals = dict((k, numpy.random.random(s).astype('float32')) for k, s in zip(keys, shapes))
            print zip(keys, shapes)
            self.one_try(vals)
        
    def one_try(self, vars):
        timeit(lambda: self.empty(vars), "baseline")
        ref = self.numpy(**vars)
        timeit(lambda: self.numpy(**vars), "numpy   ")
        # to cache the expression, since we don't want to time the compile
        try:
            res = self.numexpr(vars)
            assert (res == ref).all()
            timeit(lambda: self.numexpr(vars), "numexpr ")
        except Exception, e:
            print "numexpr: error", e
            
        try:
            gvars = dict((k, gpu_ndarray.GpuNdArrayObject(v)) for k,v in vars.iteritems())
            self.ndarray = elemwise_helper(self, gvars)
            res = self.ndarray(gvars)
            assert (res == ref).all()
            timeit(lambda: self.ndarray(gvars), "ndarray ")
        except Exception, e:
            print "ndarray: error", e
        try:
            pvars = dict((k, gpuarray.to_gpu(v)) for k,v in vars.iteritems())
            res = self.pycuda(pvars)
            assert (res.get() == ref).all()
            timeit(lambda: self.pycuda(pvars), "pycuda  ")
        except Exception, e:
            print "pycuda: error", e
            import traceback
            traceback.print_exc()

series = bench("2*sin(a)-sin(2*a)+2/3.0*sin(3*a)-1/2.0*sin(4*a)+2/5.0*sin(5*a)-1/3.0*sin(6*a)+2/7.0*sin(7*a)", a=((100,), (1000,), (100, 200)))
ap1 = bench("a+1", a=((100,), (1000,), (100, 200)))
b2 = bench("2*a + 3*b", a=((100,),), b=((100,),))
b3 = bench("a**2 + b**2 + 2*a*b", a=((100,),), b=((100,),))

if __name__ == "__main__":
    print ap1.exp
    ap1.run()
#    print b2.exp
#    b2.run()
#    print b3.exp
#    b3.run()
#    print series.exp
#    series.run()
