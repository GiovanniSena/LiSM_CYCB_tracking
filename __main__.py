#!/usr/bin/env python

import sys
import lightroot
from lightroot.context import context

ascii_art = """
.__  .__       .__     __                        __     
|  | |__| ____ |  |___/  |________  ____   _____/  |_    
|  | |  |/ ___\|  |  \   __\_  __ \/  _ \ /  _ \   __\   
|  |_|  / /_/  >   Y  \  |  |  | \(  <_> |  <_> )  |     
|____/__\___  /|___|  /__|  |__|   \____/ \____/|__|     
       /_____/      \/                                \n\n"""

def main(args=None):
    l= len(sys.argv)
    print(ascii_art)
    if l >1:  
        path = sys.argv[1]
        #for path in lightroot.check_file_list(path):
        c = context.folder_context(path)
        c.run()
    else:  print("Please specify a folder to process")
    
if __name__ == "__main__":  main() 