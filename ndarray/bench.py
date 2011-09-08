import gc, time, math
import numpy
import numexpr
import theano

try:
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    from pycuda import gpuarray
except ImportError:
    pass

try:
    from gen_elemwise import ElemwiseAlgo, call_elemwise
    import pygpu_ndarray as gpu_ndarray
except ImportError:
    pass

def timeit(f, lbl):

    gc.disable()
    t = time.time()
    f()
    est = time.time() - t
    gc.enable()

    loops = max(1, int(10**math.floor(math.log(10/est, 10))))

    gc.disable()
    t = time.time()
    for _ in xrange(loops):
        f()

    print lbl, "(", loops, "loops ):", (time.time() - t)/loops, "s"
    gc.enable()

numpy_tmpl = r"""
def numpy(a, b):
    return {0}
"""
numpy_map = {
    'sin': numpy.sin
}

numexpr_tmpl = r"""
def numexpr(a, b):
    return evaluate("{0}")
"""
numexpr_map = {
    'evaluate': numexpr.evaluate
    }

theano_tmpl = r"""
exp = {0}
"""
theano_map = {
    'sin': theano.scalar.sin
}

def empty(a, b):
    return a

def elemwise_helper(kern, a, b):
    import theano
    s_a = theano.scalar.ScalarVariable(theano.scalar.Scalar(a.dtype))
    th_a = theano.tensor.TensorVariable(theano.tensor.TensorType(a.dtype,
                                                       (False,)*len(a.shape)))
    if b is not None:
        s_b = theano.scalar.ScalarVariable(theano.scalar.Scalar(b.dtype))
        th_b = theano.tensor.TensorVariable(theano.tensor.TensorType(b.dtype,
                                                        (False,)*len(b.shape)))
    theano_code = theano_tmpl.format(kern.exp)
    res = {'a':s_a, 'b':s_b}
    exec theano_code in theano_map, res
    exp = res[exp]
    comp = theano.scalar.Composite([s_a, s_b], [exp])
    alg = ElemwiseAlgo(comp)
    
    mod = SourceModule(alg.c_src_kernel([th_a, th_b], [exp], kern.__name__, len(a.shape), static=""))
    fct = mod.get_function("kernel_%s_%d"%(kern.__name__, len(a.shape)))

    def ndarray(a, b):
        if b is None:
            inputs = (a,)
        else:
            inputs = (a, b)
        call_elemwise(fct, inputs, blocks=(inputs[0].shape[-1], 1, 1))
    return ndarray

class bench(object):
    def __init__(self, exp):
        self.exp = exp
        self.empty = empty
        numpy_code = numpy_tmpl.format(exp)
        res = {}
        exec numpy_code in numpy_map, res
        self.numpy = res['numpy']
        numexpr_code = numexpr_tmpl.format(exp)
        res = {}
        exec numexpr_code in numexpr_map, res
        self.numexpr = res['numexpr']

    def run(self):
        self.one_try(numpy.random.random((10, 20)).astype('float32'), None)
        
    def one_try(self, a, b):
        timeit(lambda: self.empty(a, b), "baseline")
        timeit(lambda: self.numpy(a, b), "numpy")
        # to cache the expression, since we don't want to time the compile
        try:
            self.numexpr(a, b)
            timeit(lambda: self.numexpr(a, b), "numexpr")
        except Exception, e:
            print "numexpr: error", e
            
        try:
            ga = gpu_ndarray.GpuNdArrayObject(a)
            if b is not None:
                gb = gpu_ndarray.GpuNdArrayObject(b)
            else:
                gb = None
            self.ndarray = elemwise_helper(self, ga, gb)
            timeit(lambda: self.ndarray(ga, gb), "ndarray")
        except Exception, e:
            print "ndarray: error", e
        
        try:
            timeit(lambda: self.pycuda(a, b), "pycuda")
        except Exception, e:
            print "pycuda: error", e

series = bench("2*sin(a)-sin(2*a)+2/3.0*sin(3*a)-1/2.0*sin(4*a)+2/5.0*sin(5*a)-1/3.0*sin(6*a)+2/7.0*sin(7*a)")


if __name__ == "__main__":
    series.run()
