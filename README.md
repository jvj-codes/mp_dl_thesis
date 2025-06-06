# Monetary Policy, Wealth–Income Effects in Danish Households: Solving a Finite Horizon Lifecycle Model with Deep Learning
In this thesis, I utilise the package EconDLSolvers by Druedahl & Jøpke (2025) (see their paper at: https://drive.google.com/file/d/1txCSkwSXSQo1zl576g7MdOc2ygoK7ucK/view, or in the Papers folder.)


## WealthIncomeMP
In the WealthIncomeMP folder, there are 3 important files,
- model_funcs.py
- WealthIncomeMPModel.py
- WealthIncomeMPModel_case.py

The model_funcs.py file defines the model.
The WealthIncomeMPModel.py is the model class.
The WealthIncomeMPModel_case.py is the model class for the case study: the shock from monetary policy.

The Jupyter Notebooks show the results of the model.
- Run_DL_MP.ipynb 
- Run_DL_MP_case.ipynb 
- Run_DL_MP_case_short_term.ipynb (see this for monetary policy shock results)
