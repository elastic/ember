<img src="resources/logo.png" align="right" width="250px" height="250px">

# Endgame Malware BEnchmark for Research

The ember dataset is a collection of 1.1 million sha256 hashes from PE files that were scanned sometime in 2017. This repository makes it easy to reproducibly train the benchmark model, extend the provided feature set, or classify new PE files with the benchmark model.

This paper describes many more details about the dataset: [https://arxiv.org/abs/1804.04637](https://arxiv.org/abs/1804.04637)

## Data

Download the data here:

[https://pubdata.endgame.com/ember/ember_dataset.tar.bz2](https://pubdata.endgame.com/ember/ember_dataset.tar.bz2)

The `ember_dataset.tar.bz2` file is 1.6GB, expands to 9.2GB, and has a sha256sum of `a5603de2f34f02ab6e21df7a0f97ec4ac84ddc65caee33fb610093dd6f9e1df9`. Here's what the extracted data looks like on disk:

```
[proth@proth-mbp data]$ ls -lh ember_dataset.tar.bz2
-rw-rw-r-- 1 proth proth 1.6G Apr 11 16:49 ember_dataset.tar.bz2
[proth@proth-mbp data]$ cd ember
[proth@proth-mbp ember]$ ls -lh
total 9.2G
-rw-rw-r-- 1 proth proth 335K Apr 11 15:51 ember_model_2017.txt
-rw-rw-r-- 1 proth proth 1.6G Apr 11 15:51 test_features.jsonl
-rw-rw-r-- 1 proth proth 427M Apr 11 15:51 train_features_0.jsonl
-rw-rw-r-- 1 proth proth 1.4G Apr 11 15:51 train_features_1.jsonl
-rw-rw-r-- 1 proth proth 1.5G Apr 11 15:51 train_features_2.jsonl
-rw-rw-r-- 1 proth proth 1.4G Apr 11 15:51 train_features_3.jsonl
-rw-rw-r-- 1 proth proth 1.5G Apr 11 15:51 train_features_4.jsonl
-rw-rw-r-- 1 proth proth 1.4G Apr 11 15:51 train_features_5.jsonl
[proth@proth-mbp ember]$ head -n 1 train_features_0.jsonl
{"sha256": "0abb4fda7d5b13801d63bee53e5e256be43e141faa077a6d149874242c3f02c2", "appeared": "2006-12", "label": 0, "histogram": [45521, 13095, 12167, 12496, 12429, 11709, 11864, 12057, 12881, 11798, 11802, 11783, 12029, 12081, 11756, 12532, 11980, 11628, 11504, 11715, 11809, 12414, 11779, 11708, 11956, 11622, 11859, 11775, 11717, 11507, 11873, 11781, 12015, 11690, 11676, 11782, 11820, 11859, 12025, 11786, 11731, 11445, 11556, 11676, 12057, 11636, 11669, 11903, 12004, 11741, 11833, 12329, 11778, 11859, 11806, 11586, 11775, 11885, 11863, 12047, 11869, 12077, 11724, 12037, 13129, 11931, 12101, 12202, 11956, 12625, 11877, 11804, 11999, 11869, 11578, 11591, 11933, 12020, 11695, 11915, 12565, 11755, 11597, 12224, 11786, 11709, 12321, 12325, 11671, 11624, 11573, 11879, 11578, 11802, 12060, 11792, 11527, 12248, 11703, 11793, 12143, 12701, 12071, 11871, 12582, 12346, 12303, 11892, 12190, 12011, 11826, 12261, 12139, 11913, 11994, 12155, 13023, 13136, 11897, 12164, 12228, 11972, 11916, 11951, 12061, 12243, 12009, 12266, 12655, 12023, 11819, 12283, 11882, 12303, 11751, 11888, 11976, 12472, 11622, 13260, 11969, 12127, 11735, 12024, 11592, 11699, 11604, 11657, 11974, 11714, 11918, 11815, 11851, 11806, 11710, 11590, 11835, 11971, 11757, 11874, 11813, 11834, 11610, 11723, 11988, 11714, 11774, 12021, 11816, 11834, 11607, 11829, 11665, 11641, 11722, 11869, 11864, 11784, 11528, 11733, 11923, 11749, 11972, 11721, 11977, 11712, 11772, 11721, 11891, 11796, 11991, 12200, 12432, 11643, 11877, 12040, 11874, 11804, 11932, 12179, 11940, 11764, 11743, 11653, 11854, 11800, 12092, 12021, 11969, 11931, 11890, 11982, 11956, 11710, 11792, 12095, 11749, 11815, 11722, 11825, 11846, 11804, 11567, 11926, 11839, 11814, 11921, 11981, 11910, 11640, 11681, 12030, 12822, 12105, 12001, 12008, 12180, 11862, 11992, 11888, 12211, 12155, 11734, 11819, 12154, 11696, 12185, 11951, 12511, 12001, 11914, 11872, 12342, 12170, 12596, 22356], "byteentropy": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1898, 6, 6, 3, 1, 1, 11, 19, 32, 3, 2, 4, 7, 9, 15, 31, 1864, 11, 9, 7, 8, 15, 25, 12, 12, 8, 10, 6, 8, 12, 23, 18, 1774, 13, 12, 17, 26, 15, 51, 26, 33, 8, 10, 10, 13, 3, 8, 29, 1631, 13, 28, 19, 29, 37, 105, 53, 52, 16, 11, 6, 14, 7, 12, 15, 4564, 92, 129, 52, 121, 197, 180, 120, 196, 13, 7, 20, 39, 16, 36, 362, 8838, 197, 304, 225, 238, 327, 473, 284, 338, 21, 53, 80, 111, 36, 77, 686, 5752, 86, 226, 182, 290, 180, 965, 518, 208, 110, 180, 165, 197, 207, 217, 2805, 7344, 169, 324, 230, 860, 490, 2336, 1285, 501, 277, 259, 360, 902, 321, 566, 4256, 8104, 268, 489, 434, 1219, 633, 3454, 2183, 688, 619, 493, 607, 534, 383, 375, 6141, 7935, 297, 487, 394, 733, 455, 1048, 1313, 1035, 560, 485, 439, 564, 359, 462, 1866, 3005, 89, 238, 175, 372, 262, 590, 517, 519, 394, 226, 236, 295, 308, 307, 659, 17257, 2362, 1165, 2188, 4313, 3314, 2149, 4166, 5604, 1518, 1984, 1812, 3512, 2432, 3869, 7891, 14457, 3068, 2198, 2894, 5416, 4317, 2584, 4108, 6279, 1030, 1379, 1225, 2923, 1626, 3374, 6610, 370748, 370509, 370926, 373544, 370740, 370211, 372830, 375415, 371989, 372095, 371755, 373615, 372116, 373375, 373929, 375883], "strings": {"numstrings": 14573, "avlength": 5.972071639333013, "printabledist": [1046, 817, 877, 803, 738, 909, 831, 842, 871, 763, 796, 773, 821, 839, 959, 831, 877, 789, 824, 840, 863, 812, 887, 856, 787, 819, 849, 849, 833, 898, 852, 858, 751, 986, 859, 887, 935, 943, 904, 959, 827, 899, 772, 858, 875, 896, 879, 917, 916, 795, 823, 974, 891, 853, 910, 918, 822, 807, 825, 832, 801, 812, 826, 836, 811, 1157, 879, 957, 1111, 1611, 930, 935, 927, 1217, 867, 915, 1185, 1039, 1169, 1231, 956, 844, 1196, 1133, 1411, 1023, 850, 960, 965, 915, 853, 802, 836, 845, 804, 900], "printables": 87031, "entropy": 6.569897560341239, "paths": 3, "urls": 0, "registry": 0, "MZ": 51}, "general": {"size": 3101705, "vsize": 380928, "has_debug": 0, "exports": 0, "imports": 156, "has_relocations": 0, "has_resources": 1, "has_signature": 0, "has_tls": 0, "symbols": 0}, "header": {"coff": {"timestamp": 1124149349, "machine": "I386", "characteristics": ["CHARA_32BIT_MACHINE", "RELOCS_STRIPPED", "EXECUTABLE_IMAGE", "LINE_NUMS_STRIPPED", "LOCAL_SYMS_STRIPPED"]}, "optional": {"subsystem": "WINDOWS_GUI", "dll_characteristics": [], "magic": "PE32", "major_image_version": 0, "minor_image_version": 0, "major_linker_version": 7, "minor_linker_version": 10, "major_operating_system_version": 4, "minor_operating_system_version": 0, "major_subsystem_version": 4, "minor_subsystem_version": 0, "sizeof_code": 26624, "sizeof_headers": 1024, "sizeof_heap_commit": 4096}}, "section": {"entry": ".text", "sections": [{"name": ".text", "size": 26624, "entropy": 6.532239617101003, "vsize": 26134, "props": ["CNT_CODE", "MEM_EXECUTE", "MEM_READ"]}, {"name": ".rdata", "size": 6656, "entropy": 5.433081641309689, "vsize": 6216, "props": ["CNT_INITIALIZED_DATA", "MEM_READ"]}, {"name": ".data", "size": 512, "entropy": 1.7424160994148217, "vsize": 172468, "props": ["CNT_INITIALIZED_DATA", "MEM_READ", "MEM_WRITE"]}, {"name": ".rsro", "size": 0, "entropy": -0.0, "vsize": 135168, "props": ["CNT_UNINITIALIZED_DATA", "MEM_READ", "MEM_WRITE"]}, {"name": ".rsrc", "size": 27648, "entropy": 5.020929764194735, "vsize": 28672, "props": ["CNT_INITIALIZED_DATA", "MEM_READ"]}]}, "imports": {"KERNEL32.dll": ["SetFileTime", "CompareFileTime", "SearchPathA", "GetShortPathNameA", "GetFullPathNameA", "MoveFileA", "lstrcatA", "SetCurrentDirectoryA", "GetFileAttributesA", "GetLastError", "CreateDirectoryA", "SetFileAttributesA", "Sleep", "GetTickCount", "GetFileSize", "GetModuleFileNameA", "ExitProcess", "GetCurrentProcess", "CopyFileA", "lstrcpynA", "GetCommandLineA", "GetWindowsDirectoryA", "CloseHandle", "GetUserDefaultLangID", "GetDiskFreeSpaceA", "GlobalUnlock", "GlobalLock", "GlobalAlloc", "CreateThread", "CreateProcessA", "CreateFileA", "GetTempFileNameA", "lstrcpyA", "lstrlenA", "SetEndOfFile", "UnmapViewOfFile", "MapViewOfFile", "CreateFileMappingA", "GetSystemDirectoryA", "RemoveDirectoryA", "lstrcmpA", "GetVolumeInformationA", "InterlockedExchange", "RtlUnwind", "lstrcmpiA", "GetEnvironmentVariableA", "ExpandEnvironmentStringsA", "GlobalFree", "WaitForSingleObject", "GetExitCodeProcess", "SetErrorMode", "GetModuleHandleA", "LoadLibraryA", "GetProcAddress", "FreeLibrary", "MultiByteToWideChar", "WritePrivateProfileStringA", "GetPrivateProfileStringA", "VirtualQuery", "WriteFile", "ReadFile", "SetFilePointer", "FindClose", "FindNextFileA", "FindFirstFileA", "DeleteFileA", "GetTempPathA", "MulDiv"], "USER32.dll": ["CloseClipboard", "SetClipboardData", "EmptyClipboard", "OpenClipboard", "TrackPopupMenu", "GetWindowRect", "AppendMenuA", "CreatePopupMenu", "GetSystemMetrics", "EndDialog", "SetWindowPos", "SetClassLongA", "IsWindowEnabled", "DialogBoxParamA", "LoadBitmapA", "GetClassInfoA", "SetDlgItemTextA", "GetDlgItemTextA", "MessageBoxA", "CharPrevA", "LoadCursorA", "GetWindowLongA", "GetSysColor", "CharNextA", "ExitWindowsEx", "CreateDialogParamA", "DestroyWindow", "SetTimer", "SetCursor", "IsWindowVisible", "CallWindowProcA", "GetMessagePos", "ScreenToClient", "CheckDlgButton", "RegisterClassA", "SetWindowTextA", "wsprintfA", "SetForegroundWindow", "ShowWindow", "SendMessageTimeoutA", "FindWindowExA", "IsWindow", "GetDlgItem", "SetWindowLongA", "GetClientRect", "LoadImageA", "GetDC", "EnableWindow", "PeekMessageA", "DispatchMessageA", "SendMessageA", "InvalidateRect", "PostQuitMessage"], "GDI32.dll": ["SetTextColor", "SetBkMode", "SetBkColor", "CreateBrushIndirect", "DeleteObject", "CreateFontIndirectA", "GetDeviceCaps"], "SHELL32.dll": ["SHFileOperationA", "SHGetSpecialFolderLocation", "SHGetMalloc", "SHBrowseForFolderA", "SHGetPathFromIDListA", "ShellExecuteA"], "ADVAPI32.dll": ["RegEnumValueA", "RegSetValueExA", "RegQueryValueExA", "RegOpenKeyExA", "RegEnumKeyA", "RegDeleteValueA", "RegDeleteKeyA", "RegCloseKey", "RegCreateKeyExA"], "COMCTL32.dll": ["ImageList_AddMasked", "ImageList_Create", "ImageList_Destroy", ""], "ole32.dll": ["OleInitialize", "CoCreateInstance", "OleUninitialize"], "VERSION.dll": ["VerQueryValueA", "GetFileVersionInfoA", "GetFileVersionInfoSizeA"], "snmpapi.dll": ["SnmpUtilOidCpy", "SnmpUtilOidNCmp", "SnmpUtilVarBindFree"]}, "exports": []}
```

The resources directory in this repo also contains an example of the raw features that are provided in the dataset.


## Installation

Ember requires Python 3. The ember model was built and the [Jupyter notebook](http://jupyter.org/) run with Python 3.6 in a [conda](https://conda.io/miniconda.html) environment defined by `environment.yml`. You can reproduce this environment by running:

```
conda env create -f environment.yml -n emberenv
source activate emberenv
pip install lief==0.8.3.post3
python setup.py install
```

If you don't need the extra packages to run the Jupyter notebook, then you can use the `environment_minimal.yml` file to install:

```
conda env create -f environment_minimal.yml -n emberenv
source activate emberenv
pip install lief==0.8.3.post3
python setup.py install
```

This minimal installation can also be achieved using pip:

```
pip install -r requirements.txt
python setup.py install
```

## Scripts

The `train_ember.py` script simplifies the model training process. It will vectorize the ember features if necessary and then train the LightGBM model.

```
python train_ember.py [/path/to/dataset]
```

The `classify_binaries.py` script will return model predictions on PE files.

```
python classify_binaries.py -m [/path/to/model] BINARIES
```

## Import Usage

The raw feature data can be expanded into vectorized form on disk for model training and into metadata form. These two functions create those extra files:

```
import ember
ember.create_vectorized_features("/data/ember/")
ember.create_metadata("/data/ember/")
```

Once created, that data can be read in using convenience functions:

```
import ember
X_train, y_train, X_test, y_test = ember.read_vectorized_features("/data/ember/")
metadata_dataframe = ember.read_metadata("/data/ember/")
```

Once the data is downloaded and the ember module is installed, this simple code should reproduce the distributed ember model:

```
import ember
ember.create_vectorized_features("/data/ember/")
lgbm_model = ember.train_model("/data/ember/")
```

Once the model is trained, the ember module can be used to make a prediction on any input PE file:

```
import ember
import lightgbm as lgb
lgbm_model = lgb.Booster(model_file="/data/ember/ember_model_2017.txt")
putty_data = open("~/putty.exe", "rb").read()
print(ember.predict_sample(lgbm_model, putty_data))
```

## Citing

If you use this data in a publication please cite the following [paper](https://arxiv.org/abs/1804.04637):

```
H. Anderson and P. Roth, "EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models”, in ArXiv e-prints. Apr. 2018.

@ARTICLE{2018arXiv180404637A,
  author = {{Anderson}, H.~S. and {Roth}, P.},
  title = "{EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models}",
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {1804.04637},
  primaryClass = "cs.CR",
  keywords = {Computer Science - Cryptography and Security},
  year = 2018,
  month = apr,
  adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180404637A},
}
```
