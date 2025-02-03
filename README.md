This repository contains the code for the paper "Fair Clustering for Data
Summarization: Improved Approximation Algorithms and Complexity Insights" by
Ameet Gadekar, Aristides Gionis and Suhas Thejaswi. In Proceedings of the ACM
Web Conference (WWW 2025). The paper is currently under review. The code is
written in Python 3.7 and uses numpy, scipy, scikitlearn and pandas python
libraries.

This code is presented only for the purpose of peer-review of the paper. The
code is not intended for any other use neither for distribution, commercial or
otherwise. The code is provided as is, without any guarantees. The authors are
not responsible for any damages that may result from using the code.

If you use our implementation, please cite us as follows.

```
@inproceedings{gadekar2023fair,
    title={Fair Clustering for Data Summarization: Improved 
           Approximation Algorithms and Complexity Insights},
    author={Gadekar, Ameet and Gionis, Aristides and Thejaswi, Suhas},
    booktitle={Proceedings of the ACM Web Conference (WWW 2025)},
    year={2025},
    publisher={ACM}
}
```

### Vanilla $k$-supplier

* ```k_supplier_3apx.py``` consists of the implementation of $3$-approximation algorithm for vanilla $k$ supplier problem without fairness constraints.

### Fair $k$-supplier with disjoint facility groups

* ```fair_k_supplier_opt.py``` contains an implementation of brute-force enumeration algorithm to obtain the optimal solution. The implementation is not scalable for large problem instances.

* ```fair_k_supplier_5apx.py ``` contains an implementation of $5$-approximation algorithm based on Chen et al. (TCS 2023), which we consider as one of the baselines.

* ```fair_k_supplier_3apx.py``` contains an implementation of $3$-approximation algorithm that we propose in this paper.

### Fair $k$-supplier with intersecting facility groups

* ```div_k_supplier_5apx.py``` contains an implementation of $5$-approximation algorithm. The implementation combines the constraint-pattern enumeration of Thejaswi et al. (KDD 2022) and invokes the $5$-approximation algorithm of Chen et al. (TCS 2023) for the disjoing groups, to give a $5$-approximation for fair $k$-supplier problem with intersecting facility groups. We consider this as a baseline to compare against our algorithm.

* ```div_k_supplier_3apx.py``` contains an implementaion of $3$-approximation algorithm that we propose in this work. The implementation combines the constraint-pattern enumeration of Thejaswi et al. (KDD 2022) and invokes our $3$-approximation algorithm  for the disjoing groups as a subroutine, to give a $3$-approximation for fair $k$-supplier problem with intersecting facility groups.


### Configuration and requirements

* See ```requirements.txt``` for ```python``` packages required for running our code.

* See ```config.py``` to update global variables such as data input and results directories. 