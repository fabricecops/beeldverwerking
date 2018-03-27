import os
from dotenv import find_dotenv, load_dotenv
import sys
load_dotenv(find_dotenv())

print(os.environ)
PATH_P = os.environ['PATH_P']
os.chdir(PATH_P)
sys.path.insert(0, PATH_P)


