

Testing Recurrent Neural Networks: 

Command to run: 

> python main.py --model \<modelName> --criterion \<criteriaName> --mode \<modeName> -- output \<output file path>

where 

\<modelName> can be in {sentiment,ucf101}
  
\<criteriaName> can be in {NC,MCDC}
  
\<modeName> can be in {test,train}

For example: 

> python main.py --model sentiment --criterion MCDC --mode test --output statistics/sentimentNC.txt
