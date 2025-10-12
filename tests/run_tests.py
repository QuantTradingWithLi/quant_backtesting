import unittest, sys, os

here = os.path.dirname(__file__)
suite = unittest.defaultTestLoader.discover(start_dir=here, pattern="test_*.py")
res = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if res.wasSuccessful() else 1)
