
# RAPPOR

[RAPPOR](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42852.pdf) is an algorithm that lets us estimate statistics about a user population, while also preserving the privacy of individual users.
It's split into two components: A part where data is collected in a privacy respecting way, and a part where the aggregated information is decoded using statistical techniques.

This notebook provides a reimplementation of second component, i.e. the statistical analysis. After having collected enough user reports, this analysis can be performed to estimate how often certain values were reported by clients.

If no dataset is available yet, or you first want to test how well the algorithm works, the notebook also provides a way to automatically generate one for you.

After having performed the entire analysis, the results are presented at the bottom of the notebook, in the form of a [list](#Listed) and a [plot](#Visually). For each candidate string that was detected, we provide an estimated count for how often this value was reported. Keep in mind that it's only possible to detect values that were reported sufficiently often and that there's no way to know which user reported which value.


```python
ALLOWED_HASH_FUNCTIONS = ["md5", "sha256"]
ALLOWED_DISTRIBUTIONS = ["normal", "exponential", "uniform", "zipf1", "zipf1.5"]
```

## Settings

These are the settings for the RAPPOR algorithm itself. First, the Bloom filter that's used can be configured. The comments behind each variable show the variable name used in the original paper.


```python
num_bits = 128         # k
num_hash_functions = 2 # h
num_cohorts = 100      # m
```

Next, the probabilities for adding noise to the Bloom filter can be set:

- $f$: Probability for randomly setting a bit in the permanent randomized response (PRR)
- $p$: Probability of setting a bit to 1 in the instantenous randomized response (IRR) if it was 0 in the PRR
- $q$: Probability of setting a bit to 1 in the instantenous randomized response (IRR) if it was 1 in the PRR


```python
f = 0
p = 0.65 
q = 0.35
```

The analysis will only report strings as detected if there is sufficient evidence. This can be configured using a statistical significance level:


```python
significance_level = 0.05
```

The reported values can either be hashed using `md5` or `sha256`.
In Google's repository, `md5` is used. For generated datasets, the choice shouldn't really matter. For custom datasets, it's important to choose the same hash function that was also used for the data collection.


```python
hash_function = ALLOWED_HASH_FUNCTIONS[0]
```

After the analysis is done, a table showing the strings with the highest estimates is displayed. You can configure how many strings this table shows:


```python
num_displayed_results = 15
```

You can either automatically let this notebook generate data, or load an existing dataset.


```python
generate_data = True
```

## Data loading

### Option 1: Data Generation


```python
num_users = 1000000
num_candidates = 100
distribution = ALLOWED_DISTRIBUTIONS[0]
```

### Option 2: Loading an existing dataset

If you already have a dataset that you want to load, change this flag to `False`:

`reported_data` should then be a Python list that contains tuples.
The first element of each tuple is a numpy array that contains the reported bits. All these arrays need to have length `num_bits`. The second element is an integer that describes which cohort the respective user is assigned to.


```python
reported_data = []
```

For `num_bits = 4`, this list might look something like this:


```python
import numpy as np

reported_data_example = [
    (np.array([1, 0, 1, 0]), 4),
    (np.array([0, 1, 1, 1]), 2)
    # , …
]
```

### Candidates

If you want to check for specific values, `candidates` should be a list of them.


```python
candidates = []
```

If your dataset also contains the true counts, `true_counts` can be a list of them for the given candidate strings. This list needs to have the same length as `candidates`, and the indices must be aligned correctly, i.e. `true_counts[i]` must provide the true counts for `candidates[i]`.


```python
true_counts = []
```

If the dataset is automatically generated, `true_counts` is filled with the correct data and `candidates` defaults to all reported values.

---

# Note: The RAPPOR implementation is starting from here. Only touch this part if you know what you are doing.

## Checking the settings

If the data is not automatically generated, the variables above need to be set correctly. Here, we perform some basic sanity checks:


```python
if not generate_data:
    if len(candidates) == 0:
        raise ValueError("If the dataset is not automatically generated, "
                         "you need to supply a list of candidates")
        
    if len(reported_data) == 0:
        raise ValueError("If the dataset is not automatically generated, "
                         "you need to load the collected data")
        
    if len(true_counts) > 0 and len(true_counts) != len(candidates):
        raise ValueError("If you provide a list of true counts, there needs "
                         "to be information about every candidate string")
```


```python
if hash_function not in ALLOWED_HASH_FUNCTIONS:
    raise NotImplementedError("Unimplemented hash function %s" % hash_function)
```


```python
if distribution not in ALLOWED_DISTRIBUTIONS:
    raise NotImplementedError("Unimplemented distribution %s" % distribution)
```

## Hash function


```python
sc.addPyFile("client/rappor.py")
sc.addPyFile("client/hmac_drbg.py")
```


```python
from rappor import get_bloom_bits as get_bloom_bits_md5
from hashlib import sha256
```


```python
def get_bloom_bits_sha256(value, cohort, num_hash_functions, num_bits):
    bits = []
    
    for hi in range(num_hash_functions):
        seed = str(cohort) + str(hi)
        digest = sha256(seed + value).digest()

        bit = ord(digest[-1]) % num_bits
        bits.append(bit)

    return bits
```


```python
hash_functions = {
    "sha256": get_bloom_bits_sha256,
    "md5": get_bloom_bits_md5
}
```


```python
get_bloom_bits = hash_functions[hash_function]
```

## Data Generation (Test-Only)

### Distributions


```python
import numpy as np
from functools import partial
from scipy.stats import rv_discrete
```


```python
def sample_normal(num_users, num_candidates):
    return np.floor(np.random.normal(num_candidates / 2, num_candidates / 6, size=(num_users)))
```


```python
def sample_uniform(num_users, num_candidates):
    return np.floor(np.random.uniform(0, num_candidates, size=(num_users)))
```


```python
def sample_exponential(num_users, num_candidates):
    return np.floor(np.random.exponential(scale=num_candidates/5,
                                          size=(num_users)))
```


```python
def sample_custom_zipf(s, num_users, num_candidates):
    pdf = 1. / np.array(range(1, num_candidates))**float(s)
    pdf = pdf / pdf.sum()
    distribution = rv_discrete(name='zipf1', values=(range(len(pdf)), pdf))
    return distribution.rvs(size=num_users)

def sample_zipf(s):
    return partial(sample_custom_zipf, s)
```

While it doesn't happen often, the distributions above can generate values that are not between $0$ and $num_candidates$. In this case, we filter them out and resample new values until we have $num_users$ valid values.


```python
def filter_out_of_bounds(seq, lower, upper):
    seq = seq[seq >= lower]
    seq = seq[seq < upper]
    return seq
```


```python
def sample(num_users, num_candidates, distribution=sample_normal):
    data = distribution(num_users, num_candidates)
    data = filter_out_of_bounds(data, 0, num_candidates)
    
    while len(data) < num_users:
        additional_data = distribution(num_users - len(data), num_candidates)
        additional_data = filter_out_of_bounds(additional_data, 0, num_candidates)
        data = np.append(data, additional_data)
    
    return data
```

### Candidate Generation


```python
def generate_candidates(num_candidates):
    return ["v%d" % i for i in range(1, num_candidates + 1)]
```


```python
if len(candidates) == 0:
    candidates = generate_candidates(num_candidates)
```


```python
distribution_map = {
    "normal": sample_normal,
    "exponential": sample_exponential,
    "uniform": sample_uniform,
    "zipf1": sample_zipf(1),
    "zipf1.5": sample_zipf(1.5)
}
```


```python
used_distribution = distribution_map[distribution]
indices = sample(num_users, num_candidates, distribution=used_distribution)
```


```python
reported_values = [candidates[int(i)] for i in indices]
```

### Assignment to cohorts

We can reuse the sampling functions we create earlier! Here, all users are assigned to cohorts uniformly randomly. The same logic is used in the shield study.


```python
cohorts = map(int, sample(num_users, num_cohorts, distribution=sample_uniform))
```

### Generating user reports


```python
def build_bloom_filter((reported_value, cohort)):
    set_bits = get_bloom_bits(reported_value, cohort, num_hash_functions, num_bits)
    
    bits = np.zeros(num_bits)
    bits[set_bits] = 1
    
    return bits, cohort
```

The individual bits are flipped according to Bernoulli distributions with probabilities $f, p, q$.
Because numpy doesn't have helpers for these, we use the equivalent binomial distributions with `num_users = 1`.


```python
def bernoulli(p, size):
    return np.random.binomial(n=1, p=p, size=(size))
```


```python
def build_prr((bits, cohort)):
    randomized_bits = np.where(bernoulli(f, num_bits))[0]
    bits[randomized_bits] = bernoulli(0.5, len(randomized_bits))
    return bits, cohort
```


```python
def build_irr((bits, cohort)):
    result = np.zeros(num_bits)
    set_bits = np.where(bits == 1)[0]
    unset_bits = np.where(bits == 0)[0]
    
    result[set_bits] = bernoulli(q, len(set_bits))
    result[unset_bits] = bernoulli(p, len(unset_bits))
    
    return result, cohort
```


```python
if generate_data:
    rdd = sc.parallelize(zip(reported_values, cohorts))
    rdd = rdd.map(build_bloom_filter).map(build_prr).map(build_irr)
    reported_data = rdd.collect()
```

### True counts


```python
if generate_data:
    true_counts = np.zeros(num_candidates)
    idx, counts = np.unique(indices, return_counts=True)
    idx = map(int, idx)
    true_counts[idx] = counts
```

## Analysis

### Summing

Individual user reports are not very useful to us, instead we need to sum up how often each bit position was reported.

`total_reports_per_cohort` is a vector containing the number of reports from the individual cohorts. `bit_counts` is a matrix
where the entry `bit_counts[i, j]` tells us how often bit `j` was set in cohort `i`.


```python
bit_counts = np.zeros((num_cohorts, num_bits))
total_reports_per_cohort = np.zeros(num_cohorts)

for bits, cohort in reported_data:
    bit_counts[cohort] += bits
    total_reports_per_cohort[cohort] += 1
    
bit_counts = bit_counts.T
```

### Target values `y `


```python
def estimate_bloom_count(bit_counts, total_reports_per_cohort):
    Y = bit_counts - ((p + 0.5 * f * q - 0.5 * f * p) * total_reports_per_cohort)
    Y /= ((1 - f) * (q - p))
    return Y
```


```python
def get_target_values(bit_counts, total_reports_per_cohort):
    Y = estimate_bloom_count(bit_counts, total_reports_per_cohort)
    return (Y / total_reports_per_cohort).T.reshape(num_bits * num_cohorts)
```


```python
y = get_target_values(bit_counts, total_reports_per_cohort)
```

### Data matrix `X`


```python
def get_features(candidates):
    matrix = []

    for cohort in range(num_cohorts):
        rows = []

        for candidate in candidates:
            bits = np.zeros(num_bits)
            bits_set = get_bloom_bits(candidate, cohort, num_hash_functions, num_bits)
            bits[bits_set] = 1
            rows.append(bits)

        for row in np.array(rows).T:
            matrix.append(row)

    X = np.array(matrix)
    
    return X
```


```python
X = get_features(candidates)
```

### Fitting


```python
from scipy.optimize import nnls
```


```python
def fit(X, y):
    x0, _ = nnls(X, y)
    return x0
```


```python
params = fit(X, y)
```

### Significance test


```python
from scipy.stats import t
from numpy.linalg import inv, norm
```


```python
def get_significant_estimates(X, y, params, num_candidates, significance_level):
    bonferroni_corrected_level = significance_level / num_candidates

    predictions = X.dot(params)
    num_datapoints, num_features = X.shape
    MSE = norm(y - predictions, ord=2)**2 / (num_datapoints - num_features)

    var = MSE * inv(X.T.dot(X)).diagonal()
    sd = np.sqrt(var)
    ts = params / sd

    degrees_of_freedom = num_datapoints - 1
    p_values = np.array([2 * (1 - t.cdf(np.abs(i), degrees_of_freedom)) for i in ts])

    significant_i = np.where(p_values <= bonferroni_corrected_level)[0]
    significant = params[significant_i]

    analyzed = np.zeros(num_candidates)
    analyzed[significant_i] = significant
    estimates = analyzed * total_reports_per_cohort.sum()
    
    return estimates
```


```python
estimates = get_significant_estimates(X, y, params, num_candidates, significance_level)
```

## Presenting the results

### Listed


```python
from pandas import DataFrame
```


```python
def create_estimate_df(candidates, estimates, original):
    indices = np.argsort(estimates)[::-1]
    reported_candidates = [candidates[i] for i in indices]
    reported_estimates = np.array(estimates[indices], dtype=np.int32)
    
    columns = ["Candidate", "Estimated count"]
    
    if len(original) == len(estimates):
        reported_original = np.array(original[indices], dtype=np.int32)
        data = np.array(zip(reported_candidates, reported_estimates, reported_original))
        columns.append("Actual count")
    else:
        data = np.array(zip(reported_candidates, reported_estimates))

    df = DataFrame(data=data)
    df.columns = columns
    return df
```


```python
num_displayed_results = min(num_displayed_results, len(candidates))
create_estimate_df(candidates, estimates, true_counts).head(num_displayed_results)
```

### Visually


```python
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
plt.figure(figsize=(16, 9))
matplotlib.rcParams.update({'font.size': 19})

handles = []
labels = []

if len(true_counts) == len(estimates):
    original_bar = plt.bar(range(num_candidates), true_counts,
                           width=1., color='orange', edgecolor='darkorange', alpha=0.6)
    handles.append(original_bar)
    labels.append("True")
    
reported_bar = plt.bar(range(num_candidates), estimates,
                       width=1., color='blue', edgecolor='darkblue', alpha=0.6)
handles.append(reported_bar)
labels.append("Estimated")

plt.title("RAPPOR results")
plt.legend(handles, labels, prop={'size': 18})
plt.xlabel("Index of candidate string")
plt.ylabel("Count")
plt.show()
```
