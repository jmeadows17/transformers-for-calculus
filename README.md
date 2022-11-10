The example data correponds to the Next Equation Selection task (https://www.overleaf.com/read/vywgxbxwmhyx)

Each row of the data is a derivation comprising 6 total steps.

Any column header containing "srepr" can be ignored. These columns are for reproducing the derivations in sympy for future interventions.

Some columns are empty. These are argument columns where the corresponding rule does not need that argument. Premises for example need no arguments.

The final 5 columns (ignoring srepr) are the correct final equation followed by 4 alternative equations. The model must select between these options given the derivation.

IMPORTANT: The numbers in some argument columns correspond to equation indexes. For example, arg_51 might be 2. This corresponds to the eq_idx in that row.

QUESTION: A row is a derivation. Do we simply [SEP] between each row element? Or do we [SEP] between STEPS within the row (e.g., between eq_idx_1 and eq_idx_2)?
