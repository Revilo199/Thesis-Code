# Thesis-Code

Run `thesis_code_clean_for_david.py` to run code. The main parameter you might want to change is `n, h`. n is the hight of the grid, h is the width. 

For my thesis, I proved when n=1 and n=2, the determinant is always 1. I conjectured the same is when n=3. I proved that it does not hold for arbitrary n, h. Specifically when n>=4 and h>=4, I conjectured that it will never be >1.

The code (without modifications) will output two files:

`{n}x{h}bbtwo.png` and `{n}x{h}bbtwoconvexstore.txt` (note "bbtwo" was the final ordering of the convex modules that I used).
The first is the nice 'image' I used as an output.
The second is the visualization of all of the convex modules, in the same ordering.
