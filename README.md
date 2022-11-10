The example data correponds to the Next Equation Selection task (https://www.overleaf.com/read/vywgxbxwmhyx)

Each row of the data is a derivation comprising 6 total steps.

Any column header containing ```srepr``` can be ignored. These columns are for reproducing the derivations in sympy for future interventions.

Some columns are empty. These are argument columns where the corresponding rule does not need that argument. Premises for example need no arguments.

The final 5 columns (ignoring srepr) are the correct final equation followed by 4 alternative equations. The model must select between these 5 options given the derivation.




If you want to make sense of a row within the data:

  1. Create a df with columns up to and including ```df['eq_6']```, including srepr columns, but discarding the negative examples. 
  2. Select a row index e.g., i = 81
  3. Call ```reconstruct_derivation(df, i, True)```

This should nicely display the derivation including annotations and equations, for easy comparison with row i.



IMPORTANT: The numbers in some argument columns correspond to equation indexes. For example, ```arg_51``` might be 2. This corresponds to the eq_idx_2 in that row.

QUESTION: A row is a derivation. Do we simply ```[SEP]``` between each row element? Or do we ```[SEP]``` between STEPS within the row (e.g., between ```eq_idx_1``` and ```eq_idx_2```)?
