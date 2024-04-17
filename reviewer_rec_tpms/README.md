## Usage

To run TPMS, download Information file of reviewers ```newid_reviewer_info_v2.json```, information file of papers ```newid_submission_info_v2.json```, 
interaction file between reviewers and papers they have reviewed ```train.txt```, and prediction interaction file between reviewers and papers they are about to review ```test.txt```

then use the following command:

```python
python main.py --rev-info-path [path of newid_submission_info_v2.json] --pap-info-path [path of newid_reviewer_info_v2.json] --train-inter-path [path of train.txt] --test-inter-path [path of test.txt]
```
