# 📡📡📡📡 IDL Group Project  🛠️🛠️🛠️🛠️

## 📌 Getting Started  
1. download MSD subset & lastfm_subset:
<br></br>
http://millionsongdataset.com/pages/getting-dataset/#subset
<br></br>
http://millionsongdataset.com/sites/default/files/lastfm/lastfm_subset.zip
<br></br>
2. pull the msongsdb repo: 
```
git clone https://github.com/tbertinmahieux/MSongsDB.git
```

3. set up older env
```
# Note: gpt does not recommend us to switch to python 2.*, and instead use a more stable and compatible 3.8 
conda create -n msd python=3.8
conda activate msd
pip install tables==3.6.1 h5py numpy six
conda install -c conda-forge pytables
```

4. open the PythonSrc/hdf5_getters.py, edit the line 39:
```python
# before (python 2.* syntax): 
return tables.openFile(h5filename, mode='r')

# after (python 3.8 syntax): 
return tables.open_file(h5filename, mode='r')
```

### 📁 Project Structure  
```
/idl_deep_play
│── /DATASET
│   ├── /MillionSongSubset
│       ├── /A
│           ├── /A
│               ├── /A
│                   ├── TRAAAAW128F429D538.h5
│                   ├── ...
│               ├── /B
│                   ├── ...
│               ├── ...
│       ├── /B
│           ├── /A
│               ├── /A
│                   ├── TRBAADN128F426B7A4.h5
│                   ├── ...
│               ├── /B
│                   ├── ...
│               ├── ...
│   ├── /lastfm_subset
│── MSongsDB
│── .gitignore (you may need to add this file manually)
│── main.py
│── millionsongsubset.tar.gz
│── README.md
```

## 📜 Run the script
```python
# make sure the conda env is under the msd 
python main.py
```

## 📩 📤 To collaborate  
Create the .gitignore file with the following content: 
```
MSongsDB
DATASET
millionsongsubset.tar.gz
```
so we are not pushing mess into the github 

