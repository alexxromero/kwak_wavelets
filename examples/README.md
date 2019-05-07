# KWAK Examples

This folder contains the sample code used to generate the examples. To generate the data files, run 

    python generate_examples.py   
    
The examples are based on the distribytion of the diphoton invarient mass for spin-0 selection found in HEPData table 9.   

This will create the following files:
- **Wide.csv** : Wide bump signal on top of a diphoton background distribution.   
- **Narrow.csv** : Narrow bump signal on top of a diphoton background distribution.    
 - **BumpDip.csv** : Narrow bump and a dip on top of a diphoton background distribution.   
- **KK.csv** : Kalusa-Klein-like oscillations on top of a diphoton background distribution.   
- **Null.csv** : Diphoton background distribution.   

The data files contain the following columns:  
: M(gamma gamma) [GeV] : Diphoton invariant mass.    
: Nevents : Number of events per bin.  
: Sigma : Sigma per bin.  
: Hypothesis : Diphoton hypothesis.   
: Generating Function : ISignal added to the background hypothesis.   


