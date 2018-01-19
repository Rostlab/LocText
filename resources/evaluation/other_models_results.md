Info: runs done by simply slight modifying the sklsvm.py file to use the respective models, without using the SVM
transforms, and making the feature arrays dense. The binarizer transformer was also tried for the Bernoulli
distribution.


# sklearn.naive_bayes.GaussianNB

## With L1 feature selection

```
# class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
r_5	202	46	101	0	0	0.8145	0.6667	0.7332	0.0031	0.8145	0.6667	0.7332	0.0030
```

## WITHOUT L1 feature selection

```
# class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
r_5	177	42	126	0	0	0.8082	0.5842	0.6782	0.0028	0.8082	0.5842	0.6782	0.0028
```

---


# sklearn.naive_bayes.MultinomialNB

## With L1 feature selection

```
# class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
r_5	191	30	112	0	0	0.8643	0.6304	0.7290	0.0030	0.8643	0.6304	0.7290	0.0031
```


## WITHOUT L1 feature selection

```
# class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
r_5	196	58	107	0	0	0.7717	0.6469	0.7038	0.0030	0.7717	0.6469	0.7038	0.0030
```

---


# sklearn.naive_bayes.BernoulliNB

## NO binarizing the features

### With L1 feature selection

```
# class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
r_5	183	38	120	0	0	0.8281	0.6040	0.6985	0.0032	0.8281	0.6040	0.6985	0.0032
```

### WITHOUT L1 feature selection

```
# class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
r_5	205	65	98	0	0	0.7593	0.6766	0.7155	0.0030	0.7593	0.6766	0.7155	0.0030
```



## binarizing the features with

That is, used `sklearn.preprocessing.Binarizer` transformer.

Didn't change apparently anything. Most of the features were binary already anyway.

### With L1 feature selection

```
# class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
r_5	183	38	120	0	0	0.8281	0.6040	0.6985	0.0032	0.8281	0.6040	0.6985	0.0032
```

### WITHOUT L1 feature selection

```
# class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
r_5	205	65	98	0	0	0.7593	0.6766	0.7155	0.0029	0.7593	0.6766	0.7155	0.0029
```


----


# sklearn.linear_model.LogisticRegression

## With L1 feature selection

```
# class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
r_5	209	19	94	0	0	0.9167	0.6898	0.7872	0.0030	0.9167	0.6898	0.7872	0.0032
```

## WITHOUT L1 feature selection

```
# class	tp	fp	fn	fp_ov	fn_ov	e|P	e|R	e|F	e|F_SE	o|P	o|R	o|F	o|F_SE
r_5	189	43	114	0	0	0.8147	0.6238	0.7065	0.0032	0.8147	0.6238	0.7065	0.0032
```
