import os
import logging
import pysys.basetest

def findTool(name):
    import shutil
    varname = 'APES_' + name.upper()
    if os.getenv(varname):
        return os.getenv(varname)
    else:
        return shutil.which(name)

class ApesHelper:
    """ A mix-in class providing functionality to test APES tools. 
    
	To use this just inherit from it in your test, but be sure to
    leave the BaseTest as the last class, for example::

		from apes.apeshelper import ApesHelper
		class PySysTest(ApesHelper, pysys.basetest.BaseTest):

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apes = ApesHelper.Implementation(self)

    HELPER_MIXIN_FIELD_NAME = 'apes'

    class Implementation(object):
        def __init__(self, testObj: pysys.basetest.BaseTest):
            import shutil
            self.owner = testObj
            self.log = logging.getLogger('pysys.apes.ApesHelper')
            self.mpiexec = findTool('mpiexec')
            self.musubi = findTool('musubi')
            self.seeder = findTool('seeder')
            self.ateles = findTool('ateles')

        def runMusubi(self, np = 1, confile = 'musubi.lua', stdouterr = 'muslog',
                      environs = os.environ):
            """ Run Musubi with the given number of MPI processes. """
            return self.owner.startProcess(
                    self.mpiexec,
                    arguments = ['-np', f'{np}', self.musubi, confile],
                    environs  = environs,
                    stdouterr = (stdouterr+'.out', stdouterr+'.err') )

        def checkMusLog(self, logfile='muslog.out'):
            """ Check for successful Musubi run. """
            self.owner.assertGrep(logfile,
                                  expr = ' *^SUCCESSFUL run! *$',
                                  contains = True,
                                  abortOnError = True)

        def assertIsClose(self, file, dir = None, ref_file = None, loadtxt_args = {},
                          rtol = 1.e-10, atol=1e-05):
            """ Assert that the elementwise difference between file and ref_file
                is below the given tolerance (relative tolerance rtol, and absolute
                tolerance atol).

                file and ref_file are loaded via numpy's loadtxt method and loadtxt_args
                will be passed along.
                ref_file is a path relative to the self.owner.reference path and defaults
                to file.
                And file is relative to dir within the self.owner.output path.
                Example::

                    self.assertIsClose('pressure.res', dir = 'tracking')

                Will compare "{self.owner.output}/tracking/pressure.res" to
                "{self.owner.reference}/pressure.res".
            """
            import numpy as np
            resfile = file
            if dir:
                resfile = os.path.join(dir, file)
            if not ref_file:
                ref_file = file
            results = np.loadtxt(
                        os.path.join(self.owner.output, resfile),
                        **loadtxt_args)
            reference = np.loadtxt(
                        os.path.join(self.owner.reference, ref_file),
                        **loadtxt_args)
            isClose = np.allclose(results, reference, rtol=rtol, atol=atol)
            self.owner.assertThat("isClose", isClose=isClose, abortOnError=True,
                                  assertMessage=f"{resfile} elements within tolerance of reference")
