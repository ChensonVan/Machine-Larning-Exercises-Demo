### LIBSVM — A Library for Support Vector Machines



1. Download the libsvm source files

   [libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)

2. Add the path of libsvm to matlab search path


3. choose **compiler** in matlab

   ```
   mex -setup c++
   ```

   For MacOSX 10.11(Sierra) Users, you have to edit two files(**clang++_maci64.xml and clang_maci65.xml in /Applications/MATLAB_R2016a.app/bin/maci64/mexopts**).

   ​

   In both of these two files, you can use **grep "MacOSX10.*sdk"** to find the lines containing the string **MacOSX10.(9/10/11).sdk**. Then copy one of these lines and change to **MacOSX10.12.sdk** and save the file by using command **w !sudo tee % in vim** .

   ![](http://p1.bpimg.com/567571/f9f2dd13b78b4c1f.png)

   ​

4. In matlab, go the folder of libsvm

```
>> mex -setup c++
>> make
>> % in the folder of libsvm, I have add heart_scale.mat file as matlab format
>> load("heart_scale.mat")
>> svmtrain
>> svmpredict
```



5.  Reference

[1. LibSVM 在matlab中的使用](http://blog.csdn.net/abcjennifer/article/details/7370177)

[2. libsvm-mat在MATLAB平台下的安装](http://www.matlabsky.com/thread-11925-1-1.html)

[3. MEX cannot find a supported compiler in MATLAB R2015b after I upgraded to Xcode 8.0](https://au.mathworks.com/matlabcentral/answers/303369-mex-cannot-find-a-supported-compiler-in-matlab-r2015b-after-i-upgraded-to-xcode-8-0#answer_235135)