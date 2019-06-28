# KWAK Examples

This directory contains sample code to generate the toy examples from Ref. [1]. The toy distributions are:  
1. A narrow Gaussian bump  
2. A wide Gaussian bump   
3. A  bump-dip combination   
4. An oscillatory pattern with a shifted starting point   

To generate the data files, run:  

    python generate_examples.py    
    
The data files contain the following columns:    
:: Nevents ::                      Number of events per bin  
:: Hypothesis ::                 Flat background hypothesis   
:: Generating Function ::  Signal added to the flat background hypothesis 

In addition, the code to generate the Kaluza-Klein-like distribution is also provided. To generate the data file, run:

    python generate_KK_examples.py 

This will generate the file "KK.csv" with the Kaluza-Klen-like distribytion and "Null.csv" with the null hypothesis. The columns are:

:: Mgg [GeV] ::    Diphoton invariant mass  
:: Nevents ::        Number of events per bin   
:: Sigma ::           Sigma per bin  
:: Hypothesis ::   Diphoton hypothesis   

To analyze these data files with KWAK, the Demo.ipynb shows how to use the package to analyze these distributions and generate plots like the ones in Ref. [1].

### References
[1] Ben G. Lillard, Tilman Plehn, Alexis Romero, Tim M. P. Tait, _Multi-scale Mining of Kinematic Distributions with Wavelets_.

