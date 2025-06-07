## Zero-shot text classification using IBM Watson.ai



```python
!pip install wget | tail -n 1
```

    Requirement already satisfied: wget in /opt/conda/envs/Python-RT24.1/lib/python3.11/site-packages (3.2)



```python
!pip install scikit-learn | tail -n 1
!pip install "ibm-watson-machine-learning>=1.0.310" | tail -n 1
```

    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/envs/Python-RT24.1/lib/python3.11/site-packages (from scikit-learn) (2.2.0)
    Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/Python-RT24.1/lib/python3.11/site-packages (from lomond->ibm-watson-machine-learning>=1.0.310) (1.16.0)



```python
url = "https://us-south.ml.cloud.ibm.com"
apiKey = "" # api key is to be entered here
```


```python
credentials = {
    "url" : url,
    "apikey" : apiKey
}
```


```python
import os

try:
    project_id = os.environ["PROJECT_ID"]
except KeyError:
    project_id = input("Enter your project id")

project_id
```




    'f925dd0c-2679-4843-8881-9a03fcc103f5'




```python
import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.

cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='', # ibm cloudObject Key to be intered here
    ibm_auth_endpoint="https://iam.cloud.ibm.com/identity/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.direct.us-south.cloud-object-storage.appdomain.cloud')

bucket = 'sentimentalanalysis-donotdelete-pr-xwbdkhjkianjn3'
object_key = 'data.csv'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_1 = pd.read_csv(body)
df_1.head(10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sentence</th>
      <th>Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The GeoSolutions technology will leverage Bene...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>$ESI on lows, down $1.50 to $2.50 BK a real po...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>2</th>
      <td>For the last quarter of 2010 , Componenta 's n...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>According to the Finnish-Russian Chamber of Co...</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Swedish buyout firm has sold its remaining...</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>5</th>
      <td>$SPY wouldn't be surprised to see a green close</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Shell's $70 Billion BG Deal Meets Shareholder ...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SSH COMMUNICATIONS SECURITY CORP STOCK EXCHANG...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Kone 's net sales rose by some 14 % year-on-ye...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>9</th>
      <td>The Stockmann department store will have a tot...</td>
      <td>neutral</td>
    </tr>
  </tbody>
</table>
</div>




```python
def mapper_function(value):
    if value == "positive":
        return 1;
    elif value == "negative":
        return -1;
    else:
        return 0;
```


```python
df_1['Sentiment'] = df_1['Sentiment'].map(mapper_function)
df_1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sentence</th>
      <th>Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The GeoSolutions technology will leverage Bene...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>$ESI on lows, down $1.50 to $2.50 BK a real po...</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>For the last quarter of 2010 , Componenta 's n...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>According to the Finnish-Russian Chamber of Co...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Swedish buyout firm has sold its remaining...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
label_map = {
    -1 : "negative",
    0: "neutral",
    1: "postive"
}
```


```python
df_1.value_counts(['Sentiment'])
```




    Sentiment
     0           3130
     1           1852
    -1            860
    Name: count, dtype: int64




```python
from sklearn.model_selection import train_test_split

data_train, data_test, y_train, y_test = train_test_split(
    df_1['Sentence'],
    df_1['Sentiment'],
    test_size = 0.3,
    random_state = 33,
    stratify = df_1['Sentiment'])

data_train = pd.DataFrame(data_train)
data_test = pd.DataFrame(data_test)

data_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sentence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5496</th>
      <td>The aim is to convert the plants into flexible...</td>
    </tr>
    <tr>
      <th>1051</th>
      <td>( ADP News ) - Feb 4 , 2009 - Finnish broadban...</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>A total of 185 Wonderware Certified SIs are av...</td>
    </tr>
    <tr>
      <th>778</th>
      <td>The contracts comprise turnkey orders for RoRo...</td>
    </tr>
    <tr>
      <th>2470</th>
      <td>A total of 1,800,000 stock options were issued...</td>
    </tr>
  </tbody>
</table>
</div>




```python
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
print([model.name for model in ModelTypes])
```

    ['FLAN_T5_XXL', 'FLAN_UL2', 'MT0_XXL', 'GPT_NEOX', 'MPT_7B_INSTRUCT2', 'STARCODER', 'LLAMA_2_70B_CHAT', 'LLAMA_2_13B_CHAT', 'GRANITE_13B_INSTRUCT', 'GRANITE_13B_CHAT', 'FLAN_T5_XL', 'GRANITE_13B_CHAT_V2', 'GRANITE_13B_INSTRUCT_V2', 'ELYZA_JAPANESE_LLAMA_2_7B_INSTRUCT', 'MIXTRAL_8X7B_INSTRUCT_V01_Q', 'CODELLAMA_34B_INSTRUCT_HF', 'GRANITE_20B_MULTILINGUAL']



```python
model_id = ModelTypes.FLAN_T5_XXL
```


```python
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.RANDOM_SEED: 33,
    GenParams.REPETITION_PENALTY:1,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1
}
```


```python
from ibm_watson_machine_learning.foundation_models import Model

model = Model(
    model_id = model_id,
    credentials = credentials,
    params= parameters,
    project_id = project_id
)
```

    /opt/conda/envs/Python-RT24.1/lib/python3.11/site-packages/ibm_watson_machine_learning/foundation_models/utils/utils.py:273: LifecycleWarning: Model 'google/flan-t5-xxl' is in deprecated state from 2025-05-28 until 2025-07-30. IDs of alternative models: None. Further details: https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-model-lifecycle.html?context=wx&audience=wdp
      warnings.warn(default_warning_template.format(



```python
instruction = """
 determine the sentiment of the sentense.
 use either 'positive', 'negative', 'neutral'.
 use the provided example as an templeate.
 """
```


```python
zero_shot_inputs = [{"input": text} for text in data_test['Sentence']]
for i in range(3):
    print(f"THe sentence example {i+1} is: \n\t {zero_shot_inputs[i]['input']}\n")
```

    THe sentence example 1 is: 
    	 `` I 'm pleased to receive the Nomination Committee 's request and confidence , '' says Jon Risfelt .
    
    THe sentence example 2 is: 
    	 : Lietuvos Respublikos sveikatos apsaugos ministerija has awarded contract to UAB `` AFFECTO LIETUVA '' for financial systems software package .
    
    THe sentence example 3 is: 
    	 The company said production volumes so far indicate the circuit is capable of the targeted output rate of 60,000 tonnes per day , or 22 million tonnes a year .
    



```python
data_train_and_labels = data_train.copy()
data_train_and_labels['Sentiment'] = y_train
```


```python
few_shot_example = []
few_shot_examples = []
for sentence, sentiment in data_train_and_labels.groupby('Sentiment')\
                                            .apply(lambda x:x.sample(3)).values:
    few_shot_example.append(f"\tsentence:\t{sentence}\n\tsenteiment:{sentiment}\n")
few_shot_examples = [ " ".join(few_shot_example)]
```


```python
few_shot_inputs_ = [{"input": text} for text in data_test['Sentence'].values]
```


```python
results = []
for inp in few_shot_inputs_[:3]:
    results.append(model.generate(" ".join([instruction + few_shot_examples[0], inp['input']]))["results"][0])
```


```python
import json
print(json.dumps(results, indent=3))
```

    [
       {
          "generated_text": "positive",
          "generated_token_count": 1,
          "input_token_count": 421,
          "stop_reason": "max_tokens"
       },
       {
          "generated_text": "neutral",
          "generated_token_count": 1,
          "input_token_count": 443,
          "stop_reason": "max_tokens"
       },
       {
          "generated_text": "positive",
          "generated_token_count": 1,
          "input_token_count": 423,
          "stop_reason": "max_tokens"
       }
    ]



```python
y_true = [ label_map[label] for label in y_test.values[:3]]
y_true
```




    ['postive', 'postive', 'neutral']




```python
y_pred = [result['generated_text'] for result in results]
y_pred
```




    ['positive', 'neutral', 'positive']




```python

```
