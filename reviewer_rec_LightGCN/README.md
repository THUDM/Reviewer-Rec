## Task

On our self-built dataset, the paper-reviewer recommendation task was tested using the LightGCN and GF-CF method.



## Requirements

- Python version >= 3.6
- Installing the relevant package according to the requirements.txt.



## Usage

Please modify the ROOT_PATH in world.py.

To run LightGCN, use the following command:

```python
python main.py --dataset="reviewer_rec" --topks="[20,5,10,50]" --model "lgn" --gpu_id 0
```

To run GF-CF, use the following command:

```python
python main.py --dataset="reviewer_rec" --topks="[20,5,10,50]" --simple_model "gf-cf" --gpu_id 0
```
