# DNN-parser
DNN parser is a tool that allows deep neural networks to be instantiated in both tensorflow and pytorch framework from a configuration json file. 
<br>The simplicity of this approach makes the creation of DNNs for training accessible to all. 
<br>The proposed version is still being developed and updated, and has been published as a pre-print by Salvatore Contino, Luca Cruciata, Orazio Gambino and Roberto Pirrone in the paper 'IODeep: an IOD for the introduction of deep learning in the DICOM standard' available at the following link<br>
[ArXiv](https://arxiv.org/abs/2311.16163)
<br>
Please cite our work as follows if you use the parser 
```
 @article{Contino_2024,
 title={IODeep: an IOD for the introduction of deep learning in the DICOM standard},
 url={http://arxiv.org/abs/2311.16163}, DOI={10.48550/arXiv.2311.16163}, note={arXiv:2311.16163 [cs, eess]},
 number={arXiv:2311.16163}, publisher={arXiv},
 author={Contino, Salvatore and Cruciata, Luca and Gambino, Orazio and Pirrone, Roberto},
 year={2024},
 month=jan }

```

<br><br>The following sections will describe the organisation of the json to be submitted to the parser, as well as a summary of the codes describing each layer. 

# Download
Parser execuble can be download from [cloud](https://unipa-my.sharepoint.com/:f:/g/personal/salvatore_contino01_unipa_it/EgedP9xYhD9ErkVJAxvF0jwB3m6yMGA9sTn_TzO5Xf21jQ?e=zCvWKW)


## Json structure
The structure of the configuration json is very simple and requires the following keys <br>

<ul>
  <strong><li>input_shape</li></strong><br></strong>
  <strong><li>architecture</li></strong><br></strong>
  <strong><li>data_format</li></strong><br></strong>
  <strong><li>backend</li></strong><br></strong>
</ul>

To the sections listed above (mandatory) the following tags can be added as separate sections. 
<ul>
  <strong><li> convolutional </li></strong>
  <ul>
      <em><li>cn</li></em>
  </ul>
  <strong><li>kernel</li></strong>
    <ul>
      <em><li>kn</li></em>
  </ul>
  <strong><li>pooling</li></strong>
    <ul>
      <em><li>pool</li></em>
  </ul>
  <strong><li>padding</li></strong>
    <ul>
      <em><li>padding</li></em>
  </ul>
  <strong><li>padding_conv</li></strong>
    <ul>
      <em><li>padding_conv</li></em>
  </ul>
</ul>
These are necessary if we have Convolutional or Dense layers contained within the key:value architecture group 

An additional tag may also be added. 
<ul>
  <strong><li>skip_connection</li></strong><br>
</ul>
If skip connections are present in the network we have constructed. The structure of this section provides a reference to the numbering of the architecture section. In fact, skip connections involve the connection between one source layer and one target layer. For this reason, the key:value pair, reports the values of the keys that the layer pair has in the architecture section.

<br>

<details>  

  <summary>An example of json strucuture of a Unet architecture is here reported.</summary>
  
```
{
    "input_shape": {
        "input_shape0": 224,
        "input_shape1": 224,
        "input_shape2": 3,
        "label_shape": 1
    },
    "architecture": {
        "1": "doubleConv",
        "2": "Maxpool2d",
        "3": "doubleConv",
        "4": "Maxpool2d",
        "5": "doubleConv",
        "6": "Maxpool2d",
        "7": "doubleConv",
        "8": "Maxpool2d",
        "9": "ElbowUNet",
        "10": "Upsample",
        "11": "UpDoubleConv",
        "12": "Upsample",
        "13": "UpDoubleConv",
        "14": "Upsample",
        "15": "UpDoubleConv",
        "16": "Upsample",
        "17": "UpDoubleConv_out",
        "18": "cnn2d"
    },
    "convolutional": {
        "cn1": 64,
        "cn2": 128,
        "cn3": 256,
        "cn4": 512,
        "cn5": 1024,
        "cn6": 512,
        "cn7": 256,
        "cn8": 128,
        "cn9": 64, 
        "cn10": 1
    },
    "kernel": {
        "kernel1": 3,
        "kernel2": 3,
        "kernel3": 3,
        "kernel4": 3,
        "kernel5": 3,
        "kernel6": 3,
        "kernel7": 3,
        "kernel8": 3,
        "kernel9": 3,
        "kernel10": 1
    },
    "pooling": {
        "pool1": 2,
        "pool2": 2,
        "pool3": 2,
        "pool4": 2
    },
    "stride": {
        "stride": 1
    },
    "padding": {
        "padding": 1
    },
    "padding_conv":{
        "padding_conv": 0
    },
    "skip_connection": {
        "1": "16",
        "3": "14",
        "5": "12",
        "7": "10"
    },
    "data_format": {
        "data_format":"channels_last"
    },
    "backend":{
        "backend":"pytorch"
    }
}
```

</details>

## Layers dictionary
|Layers code |Layers type|
| :---:   | :---: |
| `dsc1d` | Depth Separable Convolution 1D |
| `dsc2d` | Depth Separable Convolution 2D |
| `dsc1d` | Depth Separable Convolution 1D |
| `cnn1d` | Convolutional layer 1D |
| `cnn2d` | Convolutional layer 2D |
| `lstm` | Long Short-Term Memory layer |
| `gru` | Gatered Recurrent Unit layer  |
| `Avgpool1d` |  Average Pooling 1D  |
| `Avgpool2d` |  Average Pooling 2D  |
| `Maxpool1d` |  Max Pooling 1D   |
| `Maxpool1d` |  Max Pooling 2D   |
| `Flatten` |  Flattening layer  |
| `dense` |  Average Pooling 1D  |
| `dropout` |  Dropout layer  |
| `batchNorm1d` |  Batch Normalization 1D  |
| `batchNorm2d` |  Batch Normalization 2D  |
| `relu` |  Rectified Linear Unit  |
| `prelu` | Parametric REctified Linear Unit |
| `tanh` | Hyperbolic Tangent |
| `softmax` | Softmax |
| `sigmoid` | Sigmoid  |
| `Upsample` | Upsample layers |
| `doubleConv` | 2 x [Convolutional layers 2D, Batch Normalization 2D, ReLU] |
| `ElbowUnet` | 2 x [Convolutional layers 2D, Batch Normalization 2D, ReLU] with out_channels influenced by bilinear of None upsampling |
| `UpDoubleConv` | 2 x [Convolutional layers 2D, Batch Normalization 2D, ReLU] with out_channels influenced by bilinear of None upsampling |
| `UpDoubleConv_out` | UpDoubleConv layers that terminate the architecture |
| `output` | dense layers that terminate the architecture |


## Run parser 

The following parameters must be used: 
<li><strong>'-b'</strong> or <strong>'--backend</strong>; <em>to define the backend to be used for creating the network (Tensorflow or Pytorch)</em> </li><br>
<li><strong>'-n'</strong> or <strong>'--name</strong>; <em>name of the model that will be saved</em></li><br>
<li><strong>'-j'</strong> or <strong>'--json_file</strong>; <em>string containing the json to be used to compile the network</em> </li><br>
<li><strong>'-p'</strong> or <strong>'-json_path</strong>; <em>path of the json file</em></li><br>

to run the parser the command use the following command:
```
main.dist/main.exe -p Desktop/.../architecture.json -b pytorch -n model_name
```
