пЎ:
Ѕѕ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
h
Any	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
Г
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeэout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј

StatelessWhile

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

@
StaticRegexFullMatch	
input

output
"
patternstring
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
А
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleщшelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleщшelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintџџџџџџџџџ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
А
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48ще8

false_negativesVarHandleOp*
_output_shapes
: * 

debug_namefalse_negatives/*
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0

true_positivesVarHandleOp*
_output_shapes
: *

debug_nametrue_positives/*
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0

false_positivesVarHandleOp*
_output_shapes
: * 

debug_namefalse_positives/*
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0

true_positives_1VarHandleOp*
_output_shapes
: *!

debug_nametrue_positives_1/*
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
Ї
Adam/v/dense_21/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_21/bias/*
dtype0*
shape:*%
shared_nameAdam/v/dense_21/bias
y
(Adam/v/dense_21/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_21/bias*
_output_shapes
:*
dtype0
Ї
Adam/m/dense_21/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_21/bias/*
dtype0*
shape:*%
shared_nameAdam/m/dense_21/bias
y
(Adam/m/dense_21/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_21/bias*
_output_shapes
:*
dtype0
Б
Adam/v/dense_21/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_21/kernel/*
dtype0*
shape
:*'
shared_nameAdam/v/dense_21/kernel

*Adam/v/dense_21/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_21/kernel*
_output_shapes

:*
dtype0
Б
Adam/m/dense_21/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_21/kernel/*
dtype0*
shape
:*'
shared_nameAdam/m/dense_21/kernel

*Adam/m/dense_21/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_21/kernel*
_output_shapes

:*
dtype0
Ї
Adam/v/dense_20/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_20/bias/*
dtype0*
shape:*%
shared_nameAdam/v/dense_20/bias
y
(Adam/v/dense_20/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_20/bias*
_output_shapes
:*
dtype0
Ї
Adam/m/dense_20/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_20/bias/*
dtype0*
shape:*%
shared_nameAdam/m/dense_20/bias
y
(Adam/m/dense_20/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_20/bias*
_output_shapes
:*
dtype0
Б
Adam/v/dense_20/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_20/kernel/*
dtype0*
shape
:*'
shared_nameAdam/v/dense_20/kernel

*Adam/v/dense_20/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_20/kernel*
_output_shapes

:*
dtype0
Б
Adam/m/dense_20/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_20/kernel/*
dtype0*
shape
:*'
shared_nameAdam/m/dense_20/kernel

*Adam/m/dense_20/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_20/kernel*
_output_shapes

:*
dtype0
Т
Adam/v/lstm_10/lstm_cell/biasVarHandleOp*
_output_shapes
: *.

debug_name Adam/v/lstm_10/lstm_cell/bias/*
dtype0*
shape:@*.
shared_nameAdam/v/lstm_10/lstm_cell/bias

1Adam/v/lstm_10/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/v/lstm_10/lstm_cell/bias*
_output_shapes
:@*
dtype0
Т
Adam/m/lstm_10/lstm_cell/biasVarHandleOp*
_output_shapes
: *.

debug_name Adam/m/lstm_10/lstm_cell/bias/*
dtype0*
shape:@*.
shared_nameAdam/m/lstm_10/lstm_cell/bias

1Adam/m/lstm_10/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/m/lstm_10/lstm_cell/bias*
_output_shapes
:@*
dtype0
ъ
)Adam/v/lstm_10/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *:

debug_name,*Adam/v/lstm_10/lstm_cell/recurrent_kernel/*
dtype0*
shape
:@*:
shared_name+)Adam/v/lstm_10/lstm_cell/recurrent_kernel
Ї
=Adam/v/lstm_10/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp)Adam/v/lstm_10/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0
ъ
)Adam/m/lstm_10/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *:

debug_name,*Adam/m/lstm_10/lstm_cell/recurrent_kernel/*
dtype0*
shape
:@*:
shared_name+)Adam/m/lstm_10/lstm_cell/recurrent_kernel
Ї
=Adam/m/lstm_10/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp)Adam/m/lstm_10/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0
Э
Adam/v/lstm_10/lstm_cell/kernelVarHandleOp*
_output_shapes
: *0

debug_name" Adam/v/lstm_10/lstm_cell/kernel/*
dtype0*
shape:	@*0
shared_name!Adam/v/lstm_10/lstm_cell/kernel

3Adam/v/lstm_10/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/v/lstm_10/lstm_cell/kernel*
_output_shapes
:	@*
dtype0
Э
Adam/m/lstm_10/lstm_cell/kernelVarHandleOp*
_output_shapes
: *0

debug_name" Adam/m/lstm_10/lstm_cell/kernel/*
dtype0*
shape:	@*0
shared_name!Adam/m/lstm_10/lstm_cell/kernel

3Adam/m/lstm_10/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/m/lstm_10/lstm_cell/kernel*
_output_shapes
:	@*
dtype0
І
current_learning_rateVarHandleOp*
_output_shapes
: *&

debug_namecurrent_learning_rate/*
dtype0*
shape: *&
shared_namecurrent_learning_rate
w
)current_learning_rate/Read/ReadVariableOpReadVariableOpcurrent_learning_rate*
_output_shapes
: *
dtype0

	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
­
lstm_10/lstm_cell/biasVarHandleOp*
_output_shapes
: *'

debug_namelstm_10/lstm_cell/bias/*
dtype0*
shape:@*'
shared_namelstm_10/lstm_cell/bias
}
*lstm_10/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm_10/lstm_cell/bias*
_output_shapes
:@*
dtype0
е
"lstm_10/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *3

debug_name%#lstm_10/lstm_cell/recurrent_kernel/*
dtype0*
shape
:@*3
shared_name$"lstm_10/lstm_cell/recurrent_kernel

6lstm_10/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp"lstm_10/lstm_cell/recurrent_kernel*
_output_shapes

:@*
dtype0
И
lstm_10/lstm_cell/kernelVarHandleOp*
_output_shapes
: *)

debug_namelstm_10/lstm_cell/kernel/*
dtype0*
shape:	@*)
shared_namelstm_10/lstm_cell/kernel

,lstm_10/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm_10/lstm_cell/kernel*
_output_shapes
:	@*
dtype0

dense_21/biasVarHandleOp*
_output_shapes
: *

debug_namedense_21/bias/*
dtype0*
shape:*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:*
dtype0

dense_21/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_21/kernel/*
dtype0*
shape
:* 
shared_namedense_21/kernel
s
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes

:*
dtype0

dense_20/biasVarHandleOp*
_output_shapes
: *

debug_namedense_20/bias/*
dtype0*
shape:*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes
:*
dtype0

dense_20/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_20/kernel/*
dtype0*
shape
:* 
shared_namedense_20/kernel
s
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes

:*
dtype0

serving_default_input_11Placeholder*,
_output_shapes
:џџџџџџџџџ*
dtype0*!
shape:џџџџџџџџџ
л
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_11lstm_10/lstm_cell/kernel"lstm_10/lstm_cell/recurrent_kernellstm_10/lstm_cell/biasdense_20/kerneldense_20/biasdense_21/kerneldense_21/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1316958

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*в>
valueШ>BХ> BО>
л
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
С
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
І
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
І
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias*
5
.0
/1
02
$3
%4
,5
-6*
5
.0
/1
02
$3
%4
,5
-6*
* 
А
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

6trace_0
7trace_1* 

8trace_0
9trace_1* 
* 

:
_variables
;_iterations
<_current_learning_rate
=_index_dict
>
_momentums
?_velocities
@_update_step_xla*

Aserving_default* 
* 
* 
* 

Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Gtrace_0* 

Htrace_0* 

.0
/1
02*

.0
/1
02*
* 


Istates
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_3* 
6
Strace_0
Ttrace_1
Utrace_2
Vtrace_3* 
* 
у
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]_random_generator
^
state_size

.kernel
/recurrent_kernel
0bias*
* 

$0
%1*

$0
%1*
* 

_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

dtrace_0* 

etrace_0* 
_Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_20/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

,0
-1*

,0
-1*
* 

fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

ktrace_0* 

ltrace_0* 
_Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_21/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUElstm_10/lstm_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"lstm_10/lstm_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUElstm_10/lstm_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*
 
m0
n1
o2
p3*
* 
* 
* 
* 
* 
* 
r
;0
q1
r2
s3
t4
u5
v6
w7
x8
y9
z10
{11
|12
}13
~14*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEcurrent_learning_rate;optimizer/_current_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
q0
s1
u2
w3
y4
{5
}6*
5
r0
t1
v2
x3
z4
|5
~6*
c
trace_0
trace_1
trace_2
trace_3
trace_4
trace_5
trace_6* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

.0
/1
02*

.0
/1
02*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count

_fn_kwargs*
`
	variables
	keras_api

thresholds
true_positives
false_positives*
`
	variables
	keras_api

thresholds
true_positives
false_negatives*
jd
VARIABLE_VALUEAdam/m/lstm_10/lstm_cell/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/lstm_10/lstm_cell/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)Adam/m/lstm_10/lstm_cell/recurrent_kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)Adam/v/lstm_10/lstm_cell/recurrent_kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/m/lstm_10/lstm_cell/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/lstm_10/lstm_cell/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_20/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_20/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_20/bias1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_20/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_21/kernel2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_21/kernel2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_21/bias2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_21/bias2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

	variables*
* 
ga
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
* 
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ш
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_20/kerneldense_20/biasdense_21/kerneldense_21/biaslstm_10/lstm_cell/kernel"lstm_10/lstm_cell/recurrent_kernellstm_10/lstm_cell/bias	iterationcurrent_learning_rateAdam/m/lstm_10/lstm_cell/kernelAdam/v/lstm_10/lstm_cell/kernel)Adam/m/lstm_10/lstm_cell/recurrent_kernel)Adam/v/lstm_10/lstm_cell/recurrent_kernelAdam/m/lstm_10/lstm_cell/biasAdam/v/lstm_10/lstm_cell/biasAdam/m/dense_20/kernelAdam/v/dense_20/kernelAdam/m/dense_20/biasAdam/v/dense_20/biasAdam/m/dense_21/kernelAdam/v/dense_21/kernelAdam/m/dense_21/biasAdam/v/dense_21/biastotal_1count_1totalcounttrue_positives_1false_positivestrue_positivesfalse_negativesConst*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_1318982
у
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_20/kerneldense_20/biasdense_21/kerneldense_21/biaslstm_10/lstm_cell/kernel"lstm_10/lstm_cell/recurrent_kernellstm_10/lstm_cell/bias	iterationcurrent_learning_rateAdam/m/lstm_10/lstm_cell/kernelAdam/v/lstm_10/lstm_cell/kernel)Adam/m/lstm_10/lstm_cell/recurrent_kernel)Adam/v/lstm_10/lstm_cell/recurrent_kernelAdam/m/lstm_10/lstm_cell/biasAdam/v/lstm_10/lstm_cell/biasAdam/m/dense_20/kernelAdam/v/dense_20/kernelAdam/m/dense_20/biasAdam/v/dense_20/biasAdam/m/dense_21/kernelAdam/v/dense_21/kernelAdam/m/dense_21/biasAdam/v/dense_21/biastotal_1count_1totalcounttrue_positives_1false_positivestrue_positivesfalse_negatives*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_1319084ѕЙ7
З
М
E__inference_model_10_layer_call_and_return_conditional_losses_1316855
input_11"
lstm_10_1316837:	@!
lstm_10_1316839:@
lstm_10_1316841:@"
dense_20_1316844:
dense_20_1316846:"
dense_21_1316849:
dense_21_1316851:
identityЂ dense_20/StatefulPartitionedCallЂ dense_21/StatefulPartitionedCallЂlstm_10/StatefulPartitionedCallЧ
masking_10/PartitionedCallPartitionedCallinput_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_masking_10_layer_call_and_return_conditional_losses_1315933Ђ
lstm_10/StatefulPartitionedCallStatefulPartitionedCall#masking_10/PartitionedCall:output:0lstm_10_1316837lstm_10_1316839lstm_10_1316841*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lstm_10_layer_call_and_return_conditional_losses_1316836
 dense_20/StatefulPartitionedCallStatefulPartitionedCall(lstm_10/StatefulPartitionedCall:output:0dense_20_1316844dense_20_1316846*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1316381
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_1316849dense_21_1316851*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1316397x
IdentityIdentity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall ^lstm_10/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ: : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall:V R
,
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_11:'#
!
_user_specified_name	1316837:'#
!
_user_specified_name	1316839:'#
!
_user_specified_name	1316841:'#
!
_user_specified_name	1316844:'#
!
_user_specified_name	1316846:'#
!
_user_specified_name	1316849:'#
!
_user_specified_name	1316851
З
М
E__inference_model_10_layer_call_and_return_conditional_losses_1316404
input_11"
lstm_10_1316364:	@!
lstm_10_1316366:@
lstm_10_1316368:@"
dense_20_1316382:
dense_20_1316384:"
dense_21_1316398:
dense_21_1316400:
identityЂ dense_20/StatefulPartitionedCallЂ dense_21/StatefulPartitionedCallЂlstm_10/StatefulPartitionedCallЧ
masking_10/PartitionedCallPartitionedCallinput_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_masking_10_layer_call_and_return_conditional_losses_1315933Ђ
lstm_10/StatefulPartitionedCallStatefulPartitionedCall#masking_10/PartitionedCall:output:0lstm_10_1316364lstm_10_1316366lstm_10_1316368*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lstm_10_layer_call_and_return_conditional_losses_1316363
 dense_20/StatefulPartitionedCallStatefulPartitionedCall(lstm_10/StatefulPartitionedCall:output:0dense_20_1316382dense_20_1316384*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1316381
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_1316398dense_21_1316400*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1316397x
IdentityIdentity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall ^lstm_10/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ: : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall:V R
,
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_11:'#
!
_user_specified_name	1316364:'#
!
_user_specified_name	1316366:'#
!
_user_specified_name	1316368:'#
!
_user_specified_name	1316382:'#
!
_user_specified_name	1316384:'#
!
_user_specified_name	1316398:'#
!
_user_specified_name	1316400
ѕ

*__inference_dense_21_layer_call_fn_1318763

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1316397o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:'#
!
_user_specified_name	1318757:'#
!
_user_specified_name	1318759
Њ@
Э
*__inference_gpu_lstm_with_fallback_1318555

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_splitW

zeros_likeConst*
_output_shapes
:@*
dtype0*
valueB@*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:Ъ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_7f07c2bf-ca9b-4883-99d3-8a7b5b1d9c08*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
Ю@
Э
*__inference_gpu_lstm_with_fallback_1315719

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_splitW

zeros_likeConst*
_output_shapes
:@*
dtype0*
valueB@*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:г
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_ded2654b-8cb5-4033-bb74-72cb92ca8019*
api_preferred_deviceGPU*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
Ъ
ч
=__inference___backward_gpu_lstm_with_fallback_1318556_1318732
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
::эЯЊ
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯЄ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯЈ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџХ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯ
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Є
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*b
_output_shapesP
N:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:б
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:џџџџџџџџџ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
::эЯХ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
::эЯЩ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:ј
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ђ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ѓ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ѓ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Ж
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:З
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:З
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Џ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	@Д
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:@\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: e
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@g
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:@Ъ
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::б
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:@е
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:@s
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:џџџџџџџџџt

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	@g

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:@c

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:@"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*ї
_input_shapesх
т:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::џџџџџџџџџ:џџџџџџџџџ: ::::::::: : : : *=
api_implements+)lstm_7f07c2bf-ca9b-4883-99d3-8a7b5b1d9c08*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1318731*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ:1-
+
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_namestrided_slice:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:B	>

_output_shapes
:
"
_user_specified_name
CudnnRNN:W
S
,
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	transpose:WS
+
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
ExpandDims:YU
+
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameExpandDims_1:FB

_output_shapes

:
"
_user_specified_name
concat_1:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:EA

_output_shapes
: 
'
_user_specified_nameconcat_1/axis:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_8/perm:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:C?

_output_shapes
: 
%
_user_specified_nameconcat/axis
Б
о-
!cond_while_body_1313816_rewritten&
"cond_while_cond_while_loop_counter,
(cond_while_cond_while_maximum_iterations
cond_while_placeholder
cond_while_placeholder_1
cond_while_placeholder_2
cond_while_placeholder_3
cond_while_placeholder_4#
cond_while_cond_strided_slice_0a
]cond_while_tensorarrayv2read_tensorlistgetitem_cond_tensorarrayunstack_tensorlistfromtensor_0e
acond_while_tensorarrayv2read_1_tensorlistgetitem_cond_tensorarrayunstack_1_tensorlistfromtensor_0
cond_while_matmul_kernel_0*
&cond_while_matmul_1_recurrent_kernel_0
cond_while_biasadd_bias_0H
Dcond_while_tensorlistpushback_cond_cond_while_selectv2_0_accumulatorH
Dcond_while_tensorlistpushback_1_cond_cond_while_tile_1_0_accumulator
cond_while_tensorlistpushback_2_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shape_0_accumulator
cond_while_tensorlistpushback_3_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shape_1_0_accumulator
cond_while_tensorlistpushback_4_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shape_2_0_accumulator
cond_while_tensorlistpushback_5_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shape_3_0_accumulatorH
Dcond_while_tensorlistpushback_6_cond_cond_while_tile_2_0_accumulator
cond_while_tensorlistpushback_7_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shape_0_accumulator
cond_while_tensorlistpushback_8_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shape_1_0_accumulator
cond_while_tensorlistpushback_9_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shape_2_0_accumulator
cond_while_tensorlistpushback_10_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shape_3_0_accumulatorG
Ccond_while_tensorlistpushback_11_cond_cond_while_tile_0_accumulator
cond_while_tensorlistpushback_12_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_shape_0_accumulator
cond_while_tensorlistpushback_13_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_shape_2_0_accumulatorI
Econd_while_tensorlistpushback_14_cond_cond_while_tanh_1_0_accumulatorL
Hcond_while_tensorlistpushback_15_cond_cond_while_sigmoid_2_0_accumulator
cond_while_tensorlistpushback_16_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_1_grad_shape_0_accumulator
cond_while_tensorlistpushback_17_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_1_grad_shape_1_0_accumulatorP
Lcond_while_tensorlistpushback_18_cond_cond_while_placeholder_4_0_accumulatorL
Hcond_while_tensorlistpushback_19_cond_cond_while_sigmoid_1_0_accumulatorG
Ccond_while_tensorlistpushback_20_cond_cond_while_tanh_0_accumulatorJ
Fcond_while_tensorlistpushback_21_cond_cond_while_sigmoid_0_accumulator
cond_while_tensorlistpushback_22_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_grad_shape_0_accumulator
cond_while_tensorlistpushback_23_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_grad_shape_1_0_accumulatorf
bcond_while_tensorlistpushback_24_cond_cond_while_tensorarrayv2read_tensorlistgetitem_0_accumulatorP
Lcond_while_tensorlistpushback_25_cond_cond_while_placeholder_3_0_accumulatorЦ
Сcond_while_tensorlistpushback_26_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistelementshape_0_accumulatorР
Лcond_while_tensorlistpushback_27_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistlength_0_accumulatorN
Jcond_while_tensorlistpushback_28_cond_cond_while_placeholder_0_accumulator
cond_while_identity
cond_while_identity_1
cond_while_identity_2
cond_while_identity_3
cond_while_identity_4
cond_while_identity_5
cond_while_identity_6!
cond_while_cond_strided_slice_
[cond_while_tensorarrayv2read_tensorlistgetitem_cond_tensorarrayunstack_tensorlistfromtensorc
_cond_while_tensorarrayv2read_1_tensorlistgetitem_cond_tensorarrayunstack_1_tensorlistfromtensor
cond_while_matmul_kernel(
$cond_while_matmul_1_recurrent_kernel
cond_while_biasadd_bias!
cond_while_tensorlistpushback#
cond_while_tensorlistpushback_1#
cond_while_tensorlistpushback_2#
cond_while_tensorlistpushback_3#
cond_while_tensorlistpushback_4#
cond_while_tensorlistpushback_5#
cond_while_tensorlistpushback_6#
cond_while_tensorlistpushback_7#
cond_while_tensorlistpushback_8#
cond_while_tensorlistpushback_9$
 cond_while_tensorlistpushback_10$
 cond_while_tensorlistpushback_11$
 cond_while_tensorlistpushback_12$
 cond_while_tensorlistpushback_13$
 cond_while_tensorlistpushback_14$
 cond_while_tensorlistpushback_15$
 cond_while_tensorlistpushback_16$
 cond_while_tensorlistpushback_17$
 cond_while_tensorlistpushback_18$
 cond_while_tensorlistpushback_19$
 cond_while_tensorlistpushback_20$
 cond_while_tensorlistpushback_21$
 cond_while_tensorlistpushback_22$
 cond_while_tensorlistpushback_23$
 cond_while_tensorlistpushback_24$
 cond_while_tensorlistpushback_25$
 cond_while_tensorlistpushback_26$
 cond_while_tensorlistpushback_27$
 cond_while_tensorlistpushback_28
<cond/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Р
.cond/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]cond_while_tensorarrayv2read_tensorlistgetitem_cond_tensorarrayunstack_tensorlistfromtensor_0cond_while_placeholderEcond/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
>cond/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ч
0cond/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemacond_while_tensorarrayv2read_1_tensorlistgetitem_cond_tensorarrayunstack_1_tensorlistfromtensor_0cond_while_placeholderGcond/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
 
cond/while/MatMulMatMul5cond/while/TensorArrayV2Read/TensorListGetItem:item:0cond_while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@
cond/while/MatMul_1MatMulcond_while_placeholder_3&cond_while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@
cond/while/addAddV2cond/while/MatMul:product:0cond/while/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@~
cond/while/BiasAddBiasAddcond/while/add:z:0cond_while_biasadd_bias_0*
T0*'
_output_shapes
:џџџџџџџџџ@\
cond/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
cond/while/splitSplit#cond/while/split/split_dim:output:0cond/while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitj
cond/while/SigmoidSigmoidcond/while/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџl
cond/while/Sigmoid_1Sigmoidcond/while/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ{
cond/while/mulMulcond/while/Sigmoid_1:y:0cond_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџd
cond/while/TanhTanhcond/while/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџv
cond/while/mul_1Mulcond/while/Sigmoid:y:0cond/while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџu
cond/while/add_1AddV2cond/while/mul:z:0cond/while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџl
cond/while/Sigmoid_2Sigmoidcond/while/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџa
cond/while/Tanh_1Tanhcond/while/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџz
cond/while/mul_2Mulcond/while/Sigmoid_2:y:0cond/while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџj
cond/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      І
cond/while/TileTile7cond/while/TensorArrayV2Read_1/TensorListGetItem:item:0"cond/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
cond/while/SelectV2SelectV2cond/while/Tile:output:0cond/while/mul_2:z:0cond_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџl
cond/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      Њ
cond/while/Tile_1Tile7cond/while/TensorArrayV2Read_1/TensorListGetItem:item:0$cond/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџl
cond/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      Њ
cond/while/Tile_2Tile7cond/while/TensorArrayV2Read_1/TensorListGetItem:item:0$cond/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
cond/while/SelectV2_1SelectV2cond/while/Tile_1:output:0cond/while/mul_2:z:0cond_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ
cond/while/SelectV2_2SelectV2cond/while/Tile_2:output:0cond/while/add_1:z:0cond_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџw
5cond/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ќ
/cond/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemcond_while_placeholder_1>cond/while/TensorArrayV2Write/TensorListSetItem/index:output:0cond/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшвT
cond/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :o
cond/while/add_2AddV2cond_while_placeholdercond/while/add_2/y:output:0*
T0*
_output_shapes
: T
cond/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :{
cond/while/add_3AddV2"cond_while_cond_while_loop_countercond/while/add_3/y:output:0*
T0*
_output_shapes
: V
cond/while/IdentityIdentitycond/while/add_3:z:0*
T0*
_output_shapes
: l
cond/while/Identity_1Identity(cond_while_cond_while_maximum_iterations*
T0*
_output_shapes
: X
cond/while/Identity_2Identitycond/while/add_2:z:0*
T0*
_output_shapes
: 
cond/while/Identity_3Identity?cond/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: q
cond/while/Identity_4Identitycond/while/SelectV2:output:0*
T0*'
_output_shapes
:џџџџџџџџџs
cond/while/Identity_5Identitycond/while/SelectV2_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџs
cond/while/Identity_6Identitycond/while/SelectV2_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџз
cond/while/TensorListPushBackTensorListPushBackDcond_while_tensorlistpushback_cond_cond_while_selectv2_0_accumulatorcond/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшвз
cond/while/TensorListPushBack_1TensorListPushBackDcond_while_tensorlistpushback_1_cond_cond_while_tile_1_0_accumulatorcond/while/Tile_1:output:0*
_output_shapes
: *
element_dtype0
:щшШЕ
ccond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/ShapeShapecond/while/mul_2:z:0*
T0*
_output_shapes
::эЯС
econd/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_1Shapecond/while/SelectV2_1:output:0*
T0*
_output_shapes
::эЯќ
cond/while/TensorListPushBack_2TensorListPushBackcond_while_tensorlistpushback_2_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shape_0_accumulatorlcond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape:output:0*
_output_shapes
: *
element_dtype0:щшЯ
cond/while/TensorListPushBack_3TensorListPushBackcond_while_tensorlistpushback_3_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shape_1_0_accumulatorncond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_1:output:0*
_output_shapes
: *
element_dtype0:щшЯЛ
econd/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_2Shapecond_while_placeholder_3*
T0*
_output_shapes
::эЯС
econd/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_3Shapecond/while/SelectV2_1:output:0*
T0*
_output_shapes
::эЯ
cond/while/TensorListPushBack_4TensorListPushBackcond_while_tensorlistpushback_4_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shape_2_0_accumulatorncond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_2:output:0*
_output_shapes
: *
element_dtype0:щшЯ
cond/while/TensorListPushBack_5TensorListPushBackcond_while_tensorlistpushback_5_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shape_3_0_accumulatorncond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_3:output:0*
_output_shapes
: *
element_dtype0:щшЯз
cond/while/TensorListPushBack_6TensorListPushBackDcond_while_tensorlistpushback_6_cond_cond_while_tile_2_0_accumulatorcond/while/Tile_2:output:0*
_output_shapes
: *
element_dtype0
:щшШЕ
ccond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/ShapeShapecond/while/add_1:z:0*
T0*
_output_shapes
::эЯС
econd/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_1Shapecond/while/SelectV2_2:output:0*
T0*
_output_shapes
::эЯќ
cond/while/TensorListPushBack_7TensorListPushBackcond_while_tensorlistpushback_7_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shape_0_accumulatorlcond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape:output:0*
_output_shapes
: *
element_dtype0:щшЯ
cond/while/TensorListPushBack_8TensorListPushBackcond_while_tensorlistpushback_8_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shape_1_0_accumulatorncond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_1:output:0*
_output_shapes
: *
element_dtype0:щшЯЛ
econd/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_2Shapecond_while_placeholder_4*
T0*
_output_shapes
::эЯС
econd/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_3Shapecond/while/SelectV2_2:output:0*
T0*
_output_shapes
::эЯ
cond/while/TensorListPushBack_9TensorListPushBackcond_while_tensorlistpushback_9_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shape_2_0_accumulatorncond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_2:output:0*
_output_shapes
: *
element_dtype0:щшЯ
 cond/while/TensorListPushBack_10TensorListPushBackcond_while_tensorlistpushback_10_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shape_3_0_accumulatorncond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_3:output:0*
_output_shapes
: *
element_dtype0:щшЯе
 cond/while/TensorListPushBack_11TensorListPushBackCcond_while_tensorlistpushback_11_cond_cond_while_tile_0_accumulatorcond/while/Tile:output:0*
_output_shapes
: *
element_dtype0
:щшШГ
acond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/ShapeShapecond/while/mul_2:z:0*
T0*
_output_shapes
::эЯњ
 cond/while/TensorListPushBack_12TensorListPushBackcond_while_tensorlistpushback_12_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_shape_0_accumulatorjcond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape:output:0*
_output_shapes
: *
element_dtype0:щшЯЙ
ccond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_2Shapecond_while_placeholder_2*
T0*
_output_shapes
::эЯў
 cond/while/TensorListPushBack_13TensorListPushBackcond_while_tensorlistpushback_13_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_shape_2_0_accumulatorlcond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_2:output:0*
_output_shapes
: *
element_dtype0:щшЯд
 cond/while/TensorListPushBack_14TensorListPushBackEcond_while_tensorlistpushback_14_cond_cond_while_tanh_1_0_accumulatorcond/while/Tanh_1:y:0*
_output_shapes
: *
element_dtype0:щшвк
 cond/while/TensorListPushBack_15TensorListPushBackHcond_while_tensorlistpushback_15_cond_cond_while_sigmoid_2_0_accumulatorcond/while/Sigmoid_2:y:0*
_output_shapes
: *
element_dtype0:щшвЎ
^cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/ShapeShapecond/while/mul:z:0*
T0*
_output_shapes
::эЯВ
`cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Shape_1Shapecond/while/mul_1:z:0*
T0*
_output_shapes
::эЯє
 cond/while/TensorListPushBack_16TensorListPushBackcond_while_tensorlistpushback_16_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_1_grad_shape_0_accumulatorgcond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Shape:output:0*
_output_shapes
: *
element_dtype0:щшЯј
 cond/while/TensorListPushBack_17TensorListPushBackcond_while_tensorlistpushback_17_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_1_grad_shape_1_0_accumulatoricond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Shape_1:output:0*
_output_shapes
: *
element_dtype0:щшЯо
 cond/while/TensorListPushBack_18TensorListPushBackLcond_while_tensorlistpushback_18_cond_cond_while_placeholder_4_0_accumulatorcond_while_placeholder_4*
_output_shapes
: *
element_dtype0:щшвк
 cond/while/TensorListPushBack_19TensorListPushBackHcond_while_tensorlistpushback_19_cond_cond_while_sigmoid_1_0_accumulatorcond/while/Sigmoid_1:y:0*
_output_shapes
: *
element_dtype0:щшва
 cond/while/TensorListPushBack_20TensorListPushBackCcond_while_tensorlistpushback_20_cond_cond_while_tanh_0_accumulatorcond/while/Tanh:y:0*
_output_shapes
: *
element_dtype0:щшвж
 cond/while/TensorListPushBack_21TensorListPushBackFcond_while_tensorlistpushback_21_cond_cond_while_sigmoid_0_accumulatorcond/while/Sigmoid:y:0*
_output_shapes
: *
element_dtype0:щшвЕ
\cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/ShapeShapecond/while/MatMul:product:0*
T0*
_output_shapes
::эЯЙ
^cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Shape_1Shapecond/while/MatMul_1:product:0*
T0*
_output_shapes
::эЯ№
 cond/while/TensorListPushBack_22TensorListPushBackcond_while_tensorlistpushback_22_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_grad_shape_0_accumulatorecond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Shape:output:0*
_output_shapes
: *
element_dtype0:щшЯє
 cond/while/TensorListPushBack_23TensorListPushBackcond_while_tensorlistpushback_23_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_grad_shape_1_0_accumulatorgcond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Shape_1:output:0*
_output_shapes
: *
element_dtype0:щшЯ
 cond/while/TensorListPushBack_24TensorListPushBackbcond_while_tensorlistpushback_24_cond_cond_while_tensorarrayv2read_tensorlistgetitem_0_accumulator5cond/while/TensorArrayV2Read/TensorListGetItem:item:0*
_output_shapes
: *
element_dtype0:щшво
 cond/while/TensorListPushBack_25TensorListPushBackLcond_while_tensorlistpushback_25_cond_cond_while_placeholder_3_0_accumulatorcond_while_placeholder_3*
_output_shapes
: *
element_dtype0:щшв
cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListLengthTensorListLength]cond_while_tensorarrayv2read_tensorlistgetitem_cond_tensorarrayunstack_tensorlistfromtensor_0*
_output_shapes
: Е
cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListElementShapeTensorListElementShape]cond_while_tensorarrayv2read_tensorlistgetitem_cond_tensorarrayunstack_tensorlistfromtensor_0*
_output_shapes
:*

shape_type0к
 cond/while/TensorListPushBack_26TensorListPushBackСcond_while_tensorlistpushback_26_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistelementshape_0_accumulatorcond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListElementShape:element_shape:0*
_output_shapes
: *
element_dtype0:щшЯЧ
 cond/while/TensorListPushBack_27TensorListPushBackЛcond_while_tensorlistpushback_27_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistlength_0_accumulatorcond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListLength:length:0*
_output_shapes
: *
element_dtype0:щшЯк
 cond/while/TensorListPushBack_28TensorListPushBackJcond_while_tensorlistpushback_28_cond_cond_while_placeholder_0_accumulatorcond_while_placeholder*
_output_shapes
: *
element_dtype0:щшЯ"4
cond_while_biasadd_biascond_while_biasadd_bias_0"@
cond_while_cond_strided_slicecond_while_cond_strided_slice_0"3
cond_while_identitycond/while/Identity:output:0"7
cond_while_identity_1cond/while/Identity_1:output:0"7
cond_while_identity_2cond/while/Identity_2:output:0"7
cond_while_identity_3cond/while/Identity_3:output:0"7
cond_while_identity_4cond/while/Identity_4:output:0"7
cond_while_identity_5cond/while/Identity_5:output:0"7
cond_while_identity_6cond/while/Identity_6:output:0"N
$cond_while_matmul_1_recurrent_kernel&cond_while_matmul_1_recurrent_kernel_0"6
cond_while_matmul_kernelcond_while_matmul_kernel_0"Ф
_cond_while_tensorarrayv2read_1_tensorlistgetitem_cond_tensorarrayunstack_1_tensorlistfromtensoracond_while_tensorarrayv2read_1_tensorlistgetitem_cond_tensorarrayunstack_1_tensorlistfromtensor_0"М
[cond_while_tensorarrayv2read_tensorlistgetitem_cond_tensorarrayunstack_tensorlistfromtensor]cond_while_tensorarrayv2read_tensorlistgetitem_cond_tensorarrayunstack_tensorlistfromtensor_0"N
cond_while_tensorlistpushback-cond/while/TensorListPushBack:output_handle:0"R
cond_while_tensorlistpushback_1/cond/while/TensorListPushBack_1:output_handle:0"T
 cond_while_tensorlistpushback_100cond/while/TensorListPushBack_10:output_handle:0"T
 cond_while_tensorlistpushback_110cond/while/TensorListPushBack_11:output_handle:0"T
 cond_while_tensorlistpushback_120cond/while/TensorListPushBack_12:output_handle:0"T
 cond_while_tensorlistpushback_130cond/while/TensorListPushBack_13:output_handle:0"T
 cond_while_tensorlistpushback_140cond/while/TensorListPushBack_14:output_handle:0"T
 cond_while_tensorlistpushback_150cond/while/TensorListPushBack_15:output_handle:0"T
 cond_while_tensorlistpushback_160cond/while/TensorListPushBack_16:output_handle:0"T
 cond_while_tensorlistpushback_170cond/while/TensorListPushBack_17:output_handle:0"T
 cond_while_tensorlistpushback_180cond/while/TensorListPushBack_18:output_handle:0"T
 cond_while_tensorlistpushback_190cond/while/TensorListPushBack_19:output_handle:0"R
cond_while_tensorlistpushback_2/cond/while/TensorListPushBack_2:output_handle:0"T
 cond_while_tensorlistpushback_200cond/while/TensorListPushBack_20:output_handle:0"T
 cond_while_tensorlistpushback_210cond/while/TensorListPushBack_21:output_handle:0"T
 cond_while_tensorlistpushback_220cond/while/TensorListPushBack_22:output_handle:0"T
 cond_while_tensorlistpushback_230cond/while/TensorListPushBack_23:output_handle:0"T
 cond_while_tensorlistpushback_240cond/while/TensorListPushBack_24:output_handle:0"T
 cond_while_tensorlistpushback_250cond/while/TensorListPushBack_25:output_handle:0"T
 cond_while_tensorlistpushback_260cond/while/TensorListPushBack_26:output_handle:0"T
 cond_while_tensorlistpushback_270cond/while/TensorListPushBack_27:output_handle:0"T
 cond_while_tensorlistpushback_280cond/while/TensorListPushBack_28:output_handle:0"R
cond_while_tensorlistpushback_3/cond/while/TensorListPushBack_3:output_handle:0"R
cond_while_tensorlistpushback_4/cond/while/TensorListPushBack_4:output_handle:0"R
cond_while_tensorlistpushback_5/cond/while/TensorListPushBack_5:output_handle:0"R
cond_while_tensorlistpushback_6/cond/while/TensorListPushBack_6:output_handle:0"R
cond_while_tensorlistpushback_7/cond/while/TensorListPushBack_7:output_handle:0"R
cond_while_tensorlistpushback_8/cond/while/TensorListPushBack_8:output_handle:0"R
cond_while_tensorlistpushback_9/cond/while/TensorListPushBack_9:output_handle:0*(
_construction_contextkEagerRuntime*Б
_input_shapes
: : : : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : :	@:@:@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : :O K

_output_shapes
: 
1
_user_specified_namecond/while/loop_counter:UQ

_output_shapes
: 
7
_user_specified_namecond/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:JF

_output_shapes
: 
,
_user_specified_namecond/strided_slice:d`

_output_shapes
: 
F
_user_specified_name.,cond/TensorArrayUnstack/TensorListFromTensor:f	b

_output_shapes
: 
H
_user_specified_name0.cond/TensorArrayUnstack_1/TensorListFromTensor:G
C

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias:^Z

_output_shapes
: 
@
_user_specified_name(&cond/cond/while/SelectV2_0/accumulator:\X

_output_shapes
: 
>
_user_specified_name&$cond/cond/while/Tile_1_0/accumulator:АЋ

_output_shapes
: 

_user_specified_namexvcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_0/accumulator:В­

_output_shapes
: 

_user_specified_namezxcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_1_0/accumulator:В­

_output_shapes
: 

_user_specified_namezxcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_2_0/accumulator:В­

_output_shapes
: 

_user_specified_namezxcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_3_0/accumulator:\X

_output_shapes
: 
>
_user_specified_name&$cond/cond/while/Tile_2_0/accumulator:АЋ

_output_shapes
: 

_user_specified_namexvcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_0/accumulator:В­

_output_shapes
: 

_user_specified_namezxcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_1_0/accumulator:В­

_output_shapes
: 

_user_specified_namezxcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_2_0/accumulator:В­

_output_shapes
: 

_user_specified_namezxcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_3_0/accumulator:ZV

_output_shapes
: 
<
_user_specified_name$"cond/cond/while/Tile_0/accumulator:ЎЉ

_output_shapes
: 

_user_specified_namevtcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_0/accumulator:АЋ

_output_shapes
: 

_user_specified_namexvcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_2_0/accumulator:\X

_output_shapes
: 
>
_user_specified_name&$cond/cond/while/Tanh_1_0/accumulator:_[

_output_shapes
: 
A
_user_specified_name)'cond/cond/while/Sigmoid_2_0/accumulator:ЋІ

_output_shapes
: 

_user_specified_namesqcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Shape_0/accumulator:­Ј

_output_shapes
: 

_user_specified_nameuscond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Shape_1_0/accumulator:c_

_output_shapes
: 
E
_user_specified_name-+cond/cond/while/Placeholder_4_0/accumulator:_ [

_output_shapes
: 
A
_user_specified_name)'cond/cond/while/Sigmoid_1_0/accumulator:Z!V

_output_shapes
: 
<
_user_specified_name$"cond/cond/while/Tanh_0/accumulator:]"Y

_output_shapes
: 
?
_user_specified_name'%cond/cond/while/Sigmoid_0/accumulator:Љ#Є

_output_shapes
: 

_user_specified_nameqocond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Shape_0/accumulator:Ћ$І

_output_shapes
: 

_user_specified_namesqcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Shape_1_0/accumulator:y%u

_output_shapes
: 
[
_user_specified_nameCAcond/cond/while/TensorArrayV2Read/TensorListGetItem_0/accumulator:c&_

_output_shapes
: 
E
_user_specified_name-+cond/cond/while/Placeholder_3_0/accumulator:м'з

_output_shapes
: 
М
_user_specified_nameЃ cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListElementShape_0/accumulator:ж(б

_output_shapes
: 
Ж
_user_specified_namecond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListLength_0/accumulator:a)]

_output_shapes
: 
C
_user_specified_name+)cond/cond/while/Placeholder_0/accumulator
џ


%__inference_signature_wrapper_1316958
input_11
unknown:	@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
identityЂStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_1315040o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_11:'#
!
_user_specified_name	1316942:'#
!
_user_specified_name	1316944:'#
!
_user_specified_name	1316946:'#
!
_user_specified_name	1316948:'#
!
_user_specified_name	1316950:'#
!
_user_specified_name	1316952:'#
!
_user_specified_name	1316954
фH
З
cond_true_1313668
cond_cast_mask

cond_expanddims_init_h
cond_expanddims_1_init_c
cond_split_kernel!
cond_split_1_recurrent_kernel
cond_concat_bias
cond_cudnnrnnv3_inputs
cond_identity
cond_identity_1
cond_identity_2
cond_identity_3
cond_identity_4b
	cond/CastCastcond_cast_mask*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ\
cond/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :q
cond/SumSumcond/Cast:y:0#cond/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:џџџџџџџџџU
cond/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
cond/ExpandDims
ExpandDimscond_expanddims_init_hcond/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџW
cond/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
cond/ExpandDims_1
ExpandDimscond_expanddims_1_init_ccond/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџV
cond/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ё

cond/splitSplitcond/split/split_dim:output:0cond_split_kernel*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitX
cond/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :­
cond/split_1Splitcond/split_1/split_dim:output:0cond_split_1_recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split\
cond/zeros_likeConst*
_output_shapes
:@*
dtype0*
valueB@*    R
cond/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
cond/concatConcatV2cond/zeros_like:output:0cond_concat_biascond/concat/axis:output:0*
N*
T0*
_output_shapes	
:X
cond/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ќ
cond/split_2Splitcond/split_2/split_dim:output:0cond/concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split]

cond/ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџd
cond/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       x
cond/transpose	Transposecond/split:output:0cond/transpose/perm:output:0*
T0*
_output_shapes
:	f
cond/ReshapeReshapecond/transpose:y:0cond/Const:output:0*
T0*
_output_shapes	
:@f
cond/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       |
cond/transpose_1	Transposecond/split:output:1cond/transpose_1/perm:output:0*
T0*
_output_shapes
:	j
cond/Reshape_1Reshapecond/transpose_1:y:0cond/Const:output:0*
T0*
_output_shapes	
:@f
cond/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       |
cond/transpose_2	Transposecond/split:output:2cond/transpose_2/perm:output:0*
T0*
_output_shapes
:	j
cond/Reshape_2Reshapecond/transpose_2:y:0cond/Const:output:0*
T0*
_output_shapes	
:@f
cond/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       |
cond/transpose_3	Transposecond/split:output:3cond/transpose_3/perm:output:0*
T0*
_output_shapes
:	j
cond/Reshape_3Reshapecond/transpose_3:y:0cond/Const:output:0*
T0*
_output_shapes	
:@f
cond/transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       }
cond/transpose_4	Transposecond/split_1:output:0cond/transpose_4/perm:output:0*
T0*
_output_shapes

:j
cond/Reshape_4Reshapecond/transpose_4:y:0cond/Const:output:0*
T0*
_output_shapes	
:f
cond/transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       }
cond/transpose_5	Transposecond/split_1:output:1cond/transpose_5/perm:output:0*
T0*
_output_shapes

:j
cond/Reshape_5Reshapecond/transpose_5:y:0cond/Const:output:0*
T0*
_output_shapes	
:f
cond/transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       }
cond/transpose_6	Transposecond/split_1:output:2cond/transpose_6/perm:output:0*
T0*
_output_shapes

:j
cond/Reshape_6Reshapecond/transpose_6:y:0cond/Const:output:0*
T0*
_output_shapes	
:f
cond/transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       }
cond/transpose_7	Transposecond/split_1:output:3cond/transpose_7/perm:output:0*
T0*
_output_shapes

:j
cond/Reshape_7Reshapecond/transpose_7:y:0cond/Const:output:0*
T0*
_output_shapes	
:j
cond/Reshape_8Reshapecond/split_2:output:0cond/Const:output:0*
T0*
_output_shapes
:j
cond/Reshape_9Reshapecond/split_2:output:1cond/Const:output:0*
T0*
_output_shapes
:k
cond/Reshape_10Reshapecond/split_2:output:2cond/Const:output:0*
T0*
_output_shapes
:k
cond/Reshape_11Reshapecond/split_2:output:3cond/Const:output:0*
T0*
_output_shapes
:k
cond/Reshape_12Reshapecond/split_2:output:4cond/Const:output:0*
T0*
_output_shapes
:k
cond/Reshape_13Reshapecond/split_2:output:5cond/Const:output:0*
T0*
_output_shapes
:k
cond/Reshape_14Reshapecond/split_2:output:6cond/Const:output:0*
T0*
_output_shapes
:k
cond/Reshape_15Reshapecond/split_2:output:7cond/Const:output:0*
T0*
_output_shapes
:T
cond/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
cond/concat_1ConcatV2cond/Reshape:output:0cond/Reshape_1:output:0cond/Reshape_2:output:0cond/Reshape_3:output:0cond/Reshape_4:output:0cond/Reshape_5:output:0cond/Reshape_6:output:0cond/Reshape_7:output:0cond/Reshape_8:output:0cond/Reshape_9:output:0cond/Reshape_10:output:0cond/Reshape_11:output:0cond/Reshape_12:output:0cond/Reshape_13:output:0cond/Reshape_14:output:0cond/Reshape_15:output:0cond/concat_1/axis:output:0*
N*
T0*
_output_shapes

:
cond/CudnnRNNV3
CudnnRNNV3cond_cudnnrnnv3_inputscond/ExpandDims:output:0cond/ExpandDims_1:output:0cond/concat_1:output:0cond/Sum:output:0*
T0*a
_output_shapesO
M:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::*

time_major( k
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџd
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ї
cond/strided_sliceStridedSlicecond/CudnnRNNV3:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask|
cond/SqueezeSqueezecond/CudnnRNNV3:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
~
cond/Squeeze_1Squeezecond/CudnnRNNV3:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
W
cond/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
cond/ExpandDims_2
ExpandDimscond/Squeeze:output:0cond/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ`
cond/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @b
cond/IdentityIdentitycond/Squeeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџm
cond/Identity_1Identitycond/ExpandDims_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџd
cond/Identity_2Identitycond/Squeeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
cond/Identity_3Identitycond/Squeeze_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
cond/Identity_4Identitycond/runtime:output:0*
T0*
_output_shapes
: "'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0"+
cond_identity_2cond/Identity_2:output:0"+
cond_identity_3cond/Identity_3:output:0"+
cond_identity_4cond/Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@:џџџџџџџџџ:M I
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias:TP
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
,
а
while_body_1315110
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@v
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@o
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:џџџџџџџџџ@W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ш
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
:	@
 
_user_specified_namekernel:P	L

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@
<

_output_shapes
:@

_user_specified_namebias
§
М
D__inference_lstm_10_layer_call_and_return_conditional_losses_1318305

inputs/
read_readvariableop_resource:	@0
read_1_readvariableop_resource:@,
read_2_readvariableop_resource:@

identity_3ЂRead/ReadVariableOpЂRead_1/ReadVariableOpЂRead_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	@*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	@t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:@*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:@*
dtype0Z

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@Ж
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference_standard_lstm_1318032i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџh
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ё

Ц
while_cond_1317945
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1317945___redundant_placeholder05
1while_while_cond_1317945___redundant_placeholder15
1while_while_cond_1317945___redundant_placeholder25
1while_while_cond_1317945___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
;
С
!__inference_standard_lstm_1316563

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџP
ShapeShapetranspose:y:0*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ@^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ@d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:џџџџџџџџџ@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@* 
_read_only_resource_inputs
 *
bodyR
while_body_1316477*
condR
while_cond_1316476*`
output_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_2d8db02e-7366-49a3-aeb4-eef0d5b6c5e4*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
кЪ
ч
=__inference___backward_gpu_lstm_with_fallback_1317698_1317874
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
::эЯЊ
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯЄ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯЈ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџХ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯ
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:Ђ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:­
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*k
_output_shapesY
W:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:к
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
::эЯХ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
::эЯЩ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:ј
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ђ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ѓ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ѓ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Ж
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:З
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:З
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Џ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	@Д
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:@\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: e
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@g
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:@Ъ
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::б
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:@е
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:@|
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџt

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	@g

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:@c

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:@"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapesї
є:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ::џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::џџџџџџџџџ:џџџџџџџџџ: ::::::::: : : : *=
api_implements+)lstm_47cdb55f-aa18-46f4-a131-7cc1fe3544f6*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1317873*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ:1-
+
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_namestrided_slice:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:^Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
CudnnRNN:B	>

_output_shapes
:
"
_user_specified_name
CudnnRNN:`
\
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	transpose:WS
+
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
ExpandDims:YU
+
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameExpandDims_1:FB

_output_shapes

:
"
_user_specified_name
concat_1:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:EA

_output_shapes
: 
'
_user_specified_nameconcat_1/axis:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_8/perm:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:C?

_output_shapes
: 
%
_user_specified_nameconcat/axis
г
Д
)__inference_lstm_10_layer_call_fn_1317018

inputs
unknown:	@
	unknown_0:@
	unknown_1:@
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lstm_10_layer_call_and_return_conditional_losses_1316836o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:'#
!
_user_specified_name	1317010:'#
!
_user_specified_name	1317012:'#
!
_user_specified_name	1317014
Ю@
Э
*__inference_gpu_lstm_with_fallback_1317268

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_splitW

zeros_likeConst*
_output_shapes
:@*
dtype0*
valueB@*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:г
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_73bf78a1-4efb-49eb-a5db-1e73e2d416e1*
api_preferred_deviceGPU*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
,
а
while_body_1317946
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@v
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@o
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:џџџџџџџџџ@W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ш
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
:	@
 
_user_specified_namekernel:P	L

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@
<

_output_shapes
:@

_user_specified_namebias

М
D__inference_lstm_10_layer_call_and_return_conditional_losses_1315469

inputs/
read_readvariableop_resource:	@0
read_1_readvariableop_resource:@,
read_2_readvariableop_resource:@

identity_3ЂRead/ReadVariableOpЂRead_1/ReadVariableOpЂRead_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	@*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	@t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:@*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:@*
dtype0Z

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@Ж
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference_standard_lstm_1315196i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџh
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџ: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
дЏ

cond_true_1313668_grad_1313948?
;gradients_cond_grad_gradients_grad_ys_0_gradients_grad_ys_0?
;gradients_cond_grad_gradients_grad_ys_1_gradients_grad_ys_1?
;gradients_cond_grad_gradients_grad_ys_2_gradients_grad_ys_2?
;gradients_cond_grad_gradients_grad_ys_3_gradients_grad_ys_3?
;gradients_cond_grad_gradients_grad_ys_4_gradients_grad_ys_4K
Ggradients_cond_grad_gradients_cond_expanddims_2_grad_shape_cond_squeezeK
Ggradients_cond_grad_gradients_cond_squeeze_1_grad_shape_cond_cudnnrnnv3I
Egradients_cond_grad_gradients_cond_squeeze_grad_shape_cond_cudnnrnnv3<
8gradients_cond_grad_gradients_zeros_like_cond_cudnnrnnv3>
:gradients_cond_grad_gradients_zeros_like_1_cond_cudnnrnnv3`
\gradients_cond_grad_gradients_cond_cudnnrnnv3_grad_cudnnrnnbackpropv3_cond_cudnnrnnv3_inputsY
Ugradients_cond_grad_gradients_cond_cudnnrnnv3_grad_cudnnrnnbackpropv3_cond_expanddims[
Wgradients_cond_grad_gradients_cond_cudnnrnnv3_grad_cudnnrnnbackpropv3_cond_expanddims_1W
Sgradients_cond_grad_gradients_cond_cudnnrnnv3_grad_cudnnrnnbackpropv3_cond_concat_1R
Ngradients_cond_grad_gradients_cond_cudnnrnnv3_grad_cudnnrnnbackpropv3_cond_sumY
Ugradients_cond_grad_gradients_cond_cudnnrnnv3_grad_cudnnrnnbackpropv3_cond_cudnnrnnv3S
Ogradients_cond_grad_gradients_cond_expanddims_grad_shape_cond_expanddims_init_hW
Sgradients_cond_grad_gradients_cond_expanddims_1_grad_shape_cond_expanddims_1_init_c#
gradients_cond_grad_placeholder%
!gradients_cond_grad_placeholder_1%
!gradients_cond_grad_placeholder_2%
!gradients_cond_grad_placeholder_3%
!gradients_cond_grad_placeholder_4%
!gradients_cond_grad_placeholder_5%
!gradients_cond_grad_placeholder_6%
!gradients_cond_grad_placeholder_7%
!gradients_cond_grad_placeholder_8%
!gradients_cond_grad_placeholder_9&
"gradients_cond_grad_placeholder_10&
"gradients_cond_grad_placeholder_11&
"gradients_cond_grad_placeholder_12&
"gradients_cond_grad_placeholder_13&
"gradients_cond_grad_placeholder_14&
"gradients_cond_grad_placeholder_15&
"gradients_cond_grad_placeholder_16&
"gradients_cond_grad_placeholder_17&
"gradients_cond_grad_placeholder_18&
"gradients_cond_grad_placeholder_19&
"gradients_cond_grad_placeholder_20&
"gradients_cond_grad_placeholder_21&
"gradients_cond_grad_placeholder_22&
"gradients_cond_grad_placeholder_23&
"gradients_cond_grad_placeholder_24&
"gradients_cond_grad_placeholder_25&
"gradients_cond_grad_placeholder_26 
gradients_cond_grad_identity"
gradients_cond_grad_identity_1"
gradients_cond_grad_identity_2"
gradients_cond_grad_identity_3"
gradients_cond_grad_identity_4"
gradients_cond_grad_identity_5Ђ
'gradients/cond_grad/gradients/grad_ys_0Identity;gradients_cond_grad_gradients_grad_ys_0_gradients_grad_ys_0*
T0*'
_output_shapes
:џџџџџџџџџІ
'gradients/cond_grad/gradients/grad_ys_1Identity;gradients_cond_grad_gradients_grad_ys_1_gradients_grad_ys_1*
T0*+
_output_shapes
:џџџџџџџџџЂ
'gradients/cond_grad/gradients/grad_ys_2Identity;gradients_cond_grad_gradients_grad_ys_2_gradients_grad_ys_2*
T0*'
_output_shapes
:џџџџџџџџџЂ
'gradients/cond_grad/gradients/grad_ys_3Identity;gradients_cond_grad_gradients_grad_ys_3_gradients_grad_ys_3*
T0*'
_output_shapes
:џџџџџџџџџ
'gradients/cond_grad/gradients/grad_ys_4Identity;gradients_cond_grad_gradients_grad_ys_4_gradients_grad_ys_4*
T0*
_output_shapes
: 
Kgradients/cond_grad/gradients/cond/ExpandDims_2_grad/Shape/OptionalGetValueOptionalGetValueGgradients_cond_grad_gradients_cond_expanddims_2_grad_shape_cond_squeeze*'
_output_shapes
:џџџџџџџџџ*&
output_shapes
:џџџџџџџџџ*
output_types
2а
:gradients/cond_grad/gradients/cond/ExpandDims_2_grad/ShapeShapeXgradients/cond_grad/gradients/cond/ExpandDims_2_grad/Shape/OptionalGetValue:components:0*
T0*
_output_shapes
::эЯ№
<gradients/cond_grad/gradients/cond/ExpandDims_2_grad/ReshapeReshape0gradients/cond_grad/gradients/grad_ys_1:output:0Cgradients/cond_grad/gradients/cond/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
Hgradients/cond_grad/gradients/cond/Squeeze_1_grad/Shape/OptionalGetValueOptionalGetValueGgradients_cond_grad_gradients_cond_squeeze_1_grad_shape_cond_cudnnrnnv3*+
_output_shapes
:џџџџџџџџџ**
output_shapes
:џџџџџџџџџ*
output_types
2Ъ
7gradients/cond_grad/gradients/cond/Squeeze_1_grad/ShapeShapeUgradients/cond_grad/gradients/cond/Squeeze_1_grad/Shape/OptionalGetValue:components:0*
T0*
_output_shapes
::эЯю
9gradients/cond_grad/gradients/cond/Squeeze_1_grad/ReshapeReshape0gradients/cond_grad/gradients/grad_ys_3:output:0@gradients/cond_grad/gradients/cond/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџЬ
"gradients/cond_grad/gradients/AddNAddN0gradients/cond_grad/gradients/grad_ys_0:output:00gradients/cond_grad/gradients/grad_ys_2:output:0Egradients/cond_grad/gradients/cond/ExpandDims_2_grad/Reshape:output:0*
N*
T0*:
_class0
.,loc:@gradients/cond_grad/gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ
Fgradients/cond_grad/gradients/cond/Squeeze_grad/Shape/OptionalGetValueOptionalGetValueEgradients_cond_grad_gradients_cond_squeeze_grad_shape_cond_cudnnrnnv3*+
_output_shapes
:џџџџџџџџџ**
output_shapes
:џџџџџџџџџ*
output_types
2Ц
5gradients/cond_grad/gradients/cond/Squeeze_grad/ShapeShapeSgradients/cond_grad/gradients/cond/Squeeze_grad/Shape/OptionalGetValue:components:0*
T0*
_output_shapes
::эЯт
7gradients/cond_grad/gradients/cond/Squeeze_grad/ReshapeReshape(gradients/cond_grad/gradients/AddN:sum:0>gradients/cond_grad/gradients/cond/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџї
9gradients/cond_grad/gradients/zeros_like/OptionalGetValueOptionalGetValue8gradients_cond_grad_gradients_zeros_like_cond_cudnnrnnv3*+
_output_shapes
:џџџџџџџџџ**
output_shapes
:џџџџџџџџџ*
output_types
2Г
(gradients/cond_grad/gradients/zeros_like	ZerosLikeFgradients/cond_grad/gradients/zeros_like/OptionalGetValue:components:0*
T0*+
_output_shapes
:џџџџџџџџџе
;gradients/cond_grad/gradients/zeros_like_1/OptionalGetValueOptionalGetValue:gradients_cond_grad_gradients_zeros_like_1_cond_cudnnrnnv3*
_output_shapes
:*
output_shapes
:*
output_types
2Є
*gradients/cond_grad/gradients/zeros_like_1	ZerosLikeHgradients/cond_grad/gradients/zeros_like_1/OptionalGetValue:components:0*
T0*
_output_shapes
:Б
Vgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3/OptionalGetValueOptionalGetValueUgradients_cond_grad_gradients_cond_cudnnrnnv3_grad_cudnnrnnbackpropv3_cond_expanddims*+
_output_shapes
:џџџџџџџџџ**
output_shapes
:џџџџџџџџџ*
output_types
2Е
Xgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3/OptionalGetValue_1OptionalGetValueWgradients_cond_grad_gradients_cond_cudnnrnnv3_grad_cudnnrnnbackpropv3_cond_expanddims_1*+
_output_shapes
:џџџџџџџџџ**
output_shapes
:џџџџџџџџџ*
output_types
2
Xgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3/OptionalGetValue_2OptionalGetValueSgradients_cond_grad_gradients_cond_cudnnrnnv3_grad_cudnnrnnbackpropv3_cond_concat_1*
_output_shapes

:*
output_shapes

:*
output_types
2
Xgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3/OptionalGetValue_3OptionalGetValueNgradients_cond_grad_gradients_cond_cudnnrnnv3_grad_cudnnrnnbackpropv3_cond_sum*#
_output_shapes
:џџџџџџџџџ*"
output_shapes
:џџџџџџџџџ*
output_types
2
Xgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3/OptionalGetValue_4OptionalGetValueUgradients_cond_grad_gradients_cond_cudnnrnnv3_grad_cudnnrnnbackpropv3_cond_cudnnrnnv3*
_output_shapes
:*
output_shapes
:*
output_types
2Ћ

Egradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3CudnnRNNBackpropV3\gradients_cond_grad_gradients_cond_cudnnrnnv3_grad_cudnnrnnbackpropv3_cond_cudnnrnnv3_inputscgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3/OptionalGetValue:components:0egradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3/OptionalGetValue_1:components:0egradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3/OptionalGetValue_2:components:0egradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3/OptionalGetValue_3:components:0Fgradients/cond_grad/gradients/zeros_like/OptionalGetValue:components:0Sgradients/cond_grad/gradients/cond/Squeeze_grad/Shape/OptionalGetValue:components:0Ugradients/cond_grad/gradients/cond/Squeeze_1_grad/Shape/OptionalGetValue:components:0,gradients/cond_grad/gradients/zeros_like:y:0@gradients/cond_grad/gradients/cond/Squeeze_grad/Reshape:output:0Bgradients/cond_grad/gradients/cond/Squeeze_1_grad/Reshape:output:0Hgradients/cond_grad/gradients/zeros_like_1/OptionalGetValue:components:0egradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3/OptionalGetValue_4:components:0*
T0*b
_output_shapesP
N:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*

time_major( Х
8gradients/cond_grad/gradients/cond/ExpandDims_grad/ShapeShapeOgradients_cond_grad_gradients_cond_expanddims_grad_shape_cond_expanddims_init_h*
T0*
_output_shapes
::эЯ
:gradients/cond_grad/gradients/cond/ExpandDims_grad/ReshapeReshapeXgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:input_h_backprop:0Agradients/cond_grad/gradients/cond/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЫ
:gradients/cond_grad/gradients/cond/ExpandDims_1_grad/ShapeShapeSgradients_cond_grad_gradients_cond_expanddims_1_grad_shape_cond_expanddims_1_init_c*
T0*
_output_shapes
::эЯ
<gradients/cond_grad/gradients/cond/ExpandDims_1_grad/ReshapeReshapeXgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:input_c_backprop:0Cgradients/cond_grad/gradients/cond/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџw
5gradients/cond_grad/gradients/cond/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :|
:gradients/cond_grad/gradients/cond/concat_1_grad/mod/ConstConst*
_output_shapes
: *
dtype0*
value	B : ц
4gradients/cond_grad/gradients/cond/concat_1_grad/modFloorModCgradients/cond_grad/gradients/cond/concat_1_grad/mod/Const:output:0>gradients/cond_grad/gradients/cond/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 
6gradients/cond_grad/gradients/cond/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@
8gradients/cond_grad/gradients/cond/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:@
8gradients/cond_grad/gradients/cond/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:@
8gradients/cond_grad/gradients/cond/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:@
8gradients/cond_grad/gradients/cond/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:
8gradients/cond_grad/gradients/cond/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:
8gradients/cond_grad/gradients/cond/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:
8gradients/cond_grad/gradients/cond/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:
8gradients/cond_grad/gradients/cond/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:
8gradients/cond_grad/gradients/cond/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:
9gradients/cond_grad/gradients/cond/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:
9gradients/cond_grad/gradients/cond/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:
9gradients/cond_grad/gradients/cond/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:
9gradients/cond_grad/gradients/cond/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:
9gradients/cond_grad/gradients/cond/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:
9gradients/cond_grad/gradients/cond/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:К

=gradients/cond_grad/gradients/cond/concat_1_grad/ConcatOffsetConcatOffset8gradients/cond_grad/gradients/cond/concat_1_grad/mod:z:0?gradients/cond_grad/gradients/cond/concat_1_grad/Shape:output:0Agradients/cond_grad/gradients/cond/concat_1_grad/Shape_1:output:0Agradients/cond_grad/gradients/cond/concat_1_grad/Shape_2:output:0Agradients/cond_grad/gradients/cond/concat_1_grad/Shape_3:output:0Agradients/cond_grad/gradients/cond/concat_1_grad/Shape_4:output:0Agradients/cond_grad/gradients/cond/concat_1_grad/Shape_5:output:0Agradients/cond_grad/gradients/cond/concat_1_grad/Shape_6:output:0Agradients/cond_grad/gradients/cond/concat_1_grad/Shape_7:output:0Agradients/cond_grad/gradients/cond/concat_1_grad/Shape_8:output:0Agradients/cond_grad/gradients/cond/concat_1_grad/Shape_9:output:0Bgradients/cond_grad/gradients/cond/concat_1_grad/Shape_10:output:0Bgradients/cond_grad/gradients/cond/concat_1_grad/Shape_11:output:0Bgradients/cond_grad/gradients/cond/concat_1_grad/Shape_12:output:0Bgradients/cond_grad/gradients/cond/concat_1_grad/Shape_13:output:0Bgradients/cond_grad/gradients/cond/concat_1_grad/Shape_14:output:0Bgradients/cond_grad/gradients/cond/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::д
6gradients/cond_grad/gradients/cond/concat_1_grad/SliceSliceWgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:params_backprop:0Fgradients/cond_grad/gradients/cond/concat_1_grad/ConcatOffset:offset:0?gradients/cond_grad/gradients/cond/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:@и
8gradients/cond_grad/gradients/cond/concat_1_grad/Slice_1SliceWgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:params_backprop:0Fgradients/cond_grad/gradients/cond/concat_1_grad/ConcatOffset:offset:1Agradients/cond_grad/gradients/cond/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:@и
8gradients/cond_grad/gradients/cond/concat_1_grad/Slice_2SliceWgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:params_backprop:0Fgradients/cond_grad/gradients/cond/concat_1_grad/ConcatOffset:offset:2Agradients/cond_grad/gradients/cond/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:@и
8gradients/cond_grad/gradients/cond/concat_1_grad/Slice_3SliceWgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:params_backprop:0Fgradients/cond_grad/gradients/cond/concat_1_grad/ConcatOffset:offset:3Agradients/cond_grad/gradients/cond/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:@и
8gradients/cond_grad/gradients/cond/concat_1_grad/Slice_4SliceWgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:params_backprop:0Fgradients/cond_grad/gradients/cond/concat_1_grad/ConcatOffset:offset:4Agradients/cond_grad/gradients/cond/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:и
8gradients/cond_grad/gradients/cond/concat_1_grad/Slice_5SliceWgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:params_backprop:0Fgradients/cond_grad/gradients/cond/concat_1_grad/ConcatOffset:offset:5Agradients/cond_grad/gradients/cond/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:и
8gradients/cond_grad/gradients/cond/concat_1_grad/Slice_6SliceWgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:params_backprop:0Fgradients/cond_grad/gradients/cond/concat_1_grad/ConcatOffset:offset:6Agradients/cond_grad/gradients/cond/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:и
8gradients/cond_grad/gradients/cond/concat_1_grad/Slice_7SliceWgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:params_backprop:0Fgradients/cond_grad/gradients/cond/concat_1_grad/ConcatOffset:offset:7Agradients/cond_grad/gradients/cond/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:з
8gradients/cond_grad/gradients/cond/concat_1_grad/Slice_8SliceWgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:params_backprop:0Fgradients/cond_grad/gradients/cond/concat_1_grad/ConcatOffset:offset:8Agradients/cond_grad/gradients/cond/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:з
8gradients/cond_grad/gradients/cond/concat_1_grad/Slice_9SliceWgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:params_backprop:0Fgradients/cond_grad/gradients/cond/concat_1_grad/ConcatOffset:offset:9Agradients/cond_grad/gradients/cond/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:к
9gradients/cond_grad/gradients/cond/concat_1_grad/Slice_10SliceWgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:params_backprop:0Ggradients/cond_grad/gradients/cond/concat_1_grad/ConcatOffset:offset:10Bgradients/cond_grad/gradients/cond/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:к
9gradients/cond_grad/gradients/cond/concat_1_grad/Slice_11SliceWgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:params_backprop:0Ggradients/cond_grad/gradients/cond/concat_1_grad/ConcatOffset:offset:11Bgradients/cond_grad/gradients/cond/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:к
9gradients/cond_grad/gradients/cond/concat_1_grad/Slice_12SliceWgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:params_backprop:0Ggradients/cond_grad/gradients/cond/concat_1_grad/ConcatOffset:offset:12Bgradients/cond_grad/gradients/cond/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:к
9gradients/cond_grad/gradients/cond/concat_1_grad/Slice_13SliceWgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:params_backprop:0Ggradients/cond_grad/gradients/cond/concat_1_grad/ConcatOffset:offset:13Bgradients/cond_grad/gradients/cond/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:к
9gradients/cond_grad/gradients/cond/concat_1_grad/Slice_14SliceWgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:params_backprop:0Ggradients/cond_grad/gradients/cond/concat_1_grad/ConcatOffset:offset:14Bgradients/cond_grad/gradients/cond/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:к
9gradients/cond_grad/gradients/cond/concat_1_grad/Slice_15SliceWgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:params_backprop:0Ggradients/cond_grad/gradients/cond/concat_1_grad/ConcatOffset:offset:15Bgradients/cond_grad/gradients/cond/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:
5gradients/cond_grad/gradients/cond/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      э
7gradients/cond_grad/gradients/cond/Reshape_grad/ReshapeReshape?gradients/cond_grad/gradients/cond/concat_1_grad/Slice:output:0>gradients/cond_grad/gradients/cond/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	
7gradients/cond_grad/gradients/cond/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ѓ
9gradients/cond_grad/gradients/cond/Reshape_1_grad/ReshapeReshapeAgradients/cond_grad/gradients/cond/concat_1_grad/Slice_1:output:0@gradients/cond_grad/gradients/cond/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	
7gradients/cond_grad/gradients/cond/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ѓ
9gradients/cond_grad/gradients/cond/Reshape_2_grad/ReshapeReshapeAgradients/cond_grad/gradients/cond/concat_1_grad/Slice_2:output:0@gradients/cond_grad/gradients/cond/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	
7gradients/cond_grad/gradients/cond/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ѓ
9gradients/cond_grad/gradients/cond/Reshape_3_grad/ReshapeReshapeAgradients/cond_grad/gradients/cond/concat_1_grad/Slice_3:output:0@gradients/cond_grad/gradients/cond/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	
7gradients/cond_grad/gradients/cond/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ђ
9gradients/cond_grad/gradients/cond/Reshape_4_grad/ReshapeReshapeAgradients/cond_grad/gradients/cond/concat_1_grad/Slice_4:output:0@gradients/cond_grad/gradients/cond/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:
7gradients/cond_grad/gradients/cond/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ђ
9gradients/cond_grad/gradients/cond/Reshape_5_grad/ReshapeReshapeAgradients/cond_grad/gradients/cond/concat_1_grad/Slice_5:output:0@gradients/cond_grad/gradients/cond/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:
7gradients/cond_grad/gradients/cond/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ђ
9gradients/cond_grad/gradients/cond/Reshape_6_grad/ReshapeReshapeAgradients/cond_grad/gradients/cond/concat_1_grad/Slice_6:output:0@gradients/cond_grad/gradients/cond/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:
7gradients/cond_grad/gradients/cond/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      ђ
9gradients/cond_grad/gradients/cond/Reshape_7_grad/ReshapeReshapeAgradients/cond_grad/gradients/cond/concat_1_grad/Slice_7:output:0@gradients/cond_grad/gradients/cond/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:
7gradients/cond_grad/gradients/cond/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ю
9gradients/cond_grad/gradients/cond/Reshape_8_grad/ReshapeReshapeAgradients/cond_grad/gradients/cond/concat_1_grad/Slice_8:output:0@gradients/cond_grad/gradients/cond/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:
7gradients/cond_grad/gradients/cond/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ю
9gradients/cond_grad/gradients/cond/Reshape_9_grad/ReshapeReshapeAgradients/cond_grad/gradients/cond/concat_1_grad/Slice_9:output:0@gradients/cond_grad/gradients/cond/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:
8gradients/cond_grad/gradients/cond/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ё
:gradients/cond_grad/gradients/cond/Reshape_10_grad/ReshapeReshapeBgradients/cond_grad/gradients/cond/concat_1_grad/Slice_10:output:0Agradients/cond_grad/gradients/cond/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:
8gradients/cond_grad/gradients/cond/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ё
:gradients/cond_grad/gradients/cond/Reshape_11_grad/ReshapeReshapeBgradients/cond_grad/gradients/cond/concat_1_grad/Slice_11:output:0Agradients/cond_grad/gradients/cond/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:
8gradients/cond_grad/gradients/cond/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ё
:gradients/cond_grad/gradients/cond/Reshape_12_grad/ReshapeReshapeBgradients/cond_grad/gradients/cond/concat_1_grad/Slice_12:output:0Agradients/cond_grad/gradients/cond/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:
8gradients/cond_grad/gradients/cond/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ё
:gradients/cond_grad/gradients/cond/Reshape_13_grad/ReshapeReshapeBgradients/cond_grad/gradients/cond/concat_1_grad/Slice_13:output:0Agradients/cond_grad/gradients/cond/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:
8gradients/cond_grad/gradients/cond/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ё
:gradients/cond_grad/gradients/cond/Reshape_14_grad/ReshapeReshapeBgradients/cond_grad/gradients/cond/concat_1_grad/Slice_14:output:0Agradients/cond_grad/gradients/cond/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:
8gradients/cond_grad/gradients/cond/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:ё
:gradients/cond_grad/gradients/cond/Reshape_15_grad/ReshapeReshapeBgradients/cond_grad/gradients/cond/concat_1_grad/Slice_15:output:0Agradients/cond_grad/gradients/cond/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:
Igradients/cond_grad/gradients/cond/transpose_grad/InvertPermutation/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ш
Cgradients/cond_grad/gradients/cond/transpose_grad/InvertPermutationInvertPermutationRgradients/cond_grad/gradients/cond/transpose_grad/InvertPermutation/Const:output:0*
_output_shapes
:§
;gradients/cond_grad/gradients/cond/transpose_grad/transpose	Transpose@gradients/cond_grad/gradients/cond/Reshape_grad/Reshape:output:0Ggradients/cond_grad/gradients/cond/transpose_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
Kgradients/cond_grad/gradients/cond/transpose_1_grad/InvertPermutation/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ь
Egradients/cond_grad/gradients/cond/transpose_1_grad/InvertPermutationInvertPermutationTgradients/cond_grad/gradients/cond/transpose_1_grad/InvertPermutation/Const:output:0*
_output_shapes
:
=gradients/cond_grad/gradients/cond/transpose_1_grad/transpose	TransposeBgradients/cond_grad/gradients/cond/Reshape_1_grad/Reshape:output:0Igradients/cond_grad/gradients/cond/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
Kgradients/cond_grad/gradients/cond/transpose_2_grad/InvertPermutation/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ь
Egradients/cond_grad/gradients/cond/transpose_2_grad/InvertPermutationInvertPermutationTgradients/cond_grad/gradients/cond/transpose_2_grad/InvertPermutation/Const:output:0*
_output_shapes
:
=gradients/cond_grad/gradients/cond/transpose_2_grad/transpose	TransposeBgradients/cond_grad/gradients/cond/Reshape_2_grad/Reshape:output:0Igradients/cond_grad/gradients/cond/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
Kgradients/cond_grad/gradients/cond/transpose_3_grad/InvertPermutation/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ь
Egradients/cond_grad/gradients/cond/transpose_3_grad/InvertPermutationInvertPermutationTgradients/cond_grad/gradients/cond/transpose_3_grad/InvertPermutation/Const:output:0*
_output_shapes
:
=gradients/cond_grad/gradients/cond/transpose_3_grad/transpose	TransposeBgradients/cond_grad/gradients/cond/Reshape_3_grad/Reshape:output:0Igradients/cond_grad/gradients/cond/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
Kgradients/cond_grad/gradients/cond/transpose_4_grad/InvertPermutation/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ь
Egradients/cond_grad/gradients/cond/transpose_4_grad/InvertPermutationInvertPermutationTgradients/cond_grad/gradients/cond/transpose_4_grad/InvertPermutation/Const:output:0*
_output_shapes
:
=gradients/cond_grad/gradients/cond/transpose_4_grad/transpose	TransposeBgradients/cond_grad/gradients/cond/Reshape_4_grad/Reshape:output:0Igradients/cond_grad/gradients/cond/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
Kgradients/cond_grad/gradients/cond/transpose_5_grad/InvertPermutation/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ь
Egradients/cond_grad/gradients/cond/transpose_5_grad/InvertPermutationInvertPermutationTgradients/cond_grad/gradients/cond/transpose_5_grad/InvertPermutation/Const:output:0*
_output_shapes
:
=gradients/cond_grad/gradients/cond/transpose_5_grad/transpose	TransposeBgradients/cond_grad/gradients/cond/Reshape_5_grad/Reshape:output:0Igradients/cond_grad/gradients/cond/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
Kgradients/cond_grad/gradients/cond/transpose_6_grad/InvertPermutation/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ь
Egradients/cond_grad/gradients/cond/transpose_6_grad/InvertPermutationInvertPermutationTgradients/cond_grad/gradients/cond/transpose_6_grad/InvertPermutation/Const:output:0*
_output_shapes
:
=gradients/cond_grad/gradients/cond/transpose_6_grad/transpose	TransposeBgradients/cond_grad/gradients/cond/Reshape_6_grad/Reshape:output:0Igradients/cond_grad/gradients/cond/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
Kgradients/cond_grad/gradients/cond/transpose_7_grad/InvertPermutation/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ь
Egradients/cond_grad/gradients/cond/transpose_7_grad/InvertPermutationInvertPermutationTgradients/cond_grad/gradients/cond/transpose_7_grad/InvertPermutation/Const:output:0*
_output_shapes
:
=gradients/cond_grad/gradients/cond/transpose_7_grad/transpose	TransposeBgradients/cond_grad/gradients/cond/Reshape_7_grad/Reshape:output:0Igradients/cond_grad/gradients/cond/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:~
<gradients/cond_grad/gradients/cond/split_2_grad/concat/ConstConst*
_output_shapes
: *
dtype0*
value	B : о
6gradients/cond_grad/gradients/cond/split_2_grad/concatConcatV2Bgradients/cond_grad/gradients/cond/Reshape_8_grad/Reshape:output:0Bgradients/cond_grad/gradients/cond/Reshape_9_grad/Reshape:output:0Cgradients/cond_grad/gradients/cond/Reshape_10_grad/Reshape:output:0Cgradients/cond_grad/gradients/cond/Reshape_11_grad/Reshape:output:0Cgradients/cond_grad/gradients/cond/Reshape_12_grad/Reshape:output:0Cgradients/cond_grad/gradients/cond/Reshape_13_grad/Reshape:output:0Cgradients/cond_grad/gradients/cond/Reshape_14_grad/Reshape:output:0Cgradients/cond_grad/gradients/cond/Reshape_15_grad/Reshape:output:0Egradients/cond_grad/gradients/cond/split_2_grad/concat/Const:output:0*
N*
T0*
_output_shapes	
:|
:gradients/cond_grad/gradients/cond/split_grad/concat/ConstConst*
_output_shapes
: *
dtype0*
value	B :Т
4gradients/cond_grad/gradients/cond/split_grad/concatConcatV2?gradients/cond_grad/gradients/cond/transpose_grad/transpose:y:0Agradients/cond_grad/gradients/cond/transpose_1_grad/transpose:y:0Agradients/cond_grad/gradients/cond/transpose_2_grad/transpose:y:0Agradients/cond_grad/gradients/cond/transpose_3_grad/transpose:y:0Cgradients/cond_grad/gradients/cond/split_grad/concat/Const:output:0*
N*
T0*
_output_shapes
:	@~
<gradients/cond_grad/gradients/cond/split_1_grad/concat/ConstConst*
_output_shapes
: *
dtype0*
value	B :Ч
6gradients/cond_grad/gradients/cond/split_1_grad/concatConcatV2Agradients/cond_grad/gradients/cond/transpose_4_grad/transpose:y:0Agradients/cond_grad/gradients/cond/transpose_5_grad/transpose:y:0Agradients/cond_grad/gradients/cond/transpose_6_grad/transpose:y:0Agradients/cond_grad/gradients/cond/transpose_7_grad/transpose:y:0Egradients/cond_grad/gradients/cond/split_1_grad/concat/Const:output:0*
N*
T0*
_output_shapes

:@u
3gradients/cond_grad/gradients/cond/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :z
8gradients/cond_grad/gradients/cond/concat_grad/mod/ConstConst*
_output_shapes
: *
dtype0*
value	B : р
2gradients/cond_grad/gradients/cond/concat_grad/modFloorModAgradients/cond_grad/gradients/cond/concat_grad/mod/Const:output:0<gradients/cond_grad/gradients/cond/concat_grad/Rank:output:0*
T0*
_output_shapes
: ~
4gradients/cond_grad/gradients/cond/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@
6gradients/cond_grad/gradients/cond/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:@Ў
;gradients/cond_grad/gradients/cond/concat_grad/ConcatOffsetConcatOffset6gradients/cond_grad/gradients/cond/concat_grad/mod:z:0=gradients/cond_grad/gradients/cond/concat_grad/Shape:output:0?gradients/cond_grad/gradients/cond/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::Е
4gradients/cond_grad/gradients/cond/concat_grad/SliceSlice?gradients/cond_grad/gradients/cond/split_2_grad/concat:output:0Dgradients/cond_grad/gradients/cond/concat_grad/ConcatOffset:offset:0=gradients/cond_grad/gradients/cond/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:@Й
6gradients/cond_grad/gradients/cond/concat_grad/Slice_1Slice?gradients/cond_grad/gradients/cond/split_2_grad/concat:output:0Dgradients/cond_grad/gradients/cond/concat_grad/ConcatOffset:offset:1?gradients/cond_grad/gradients/cond/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:@
gradients/cond_grad/IdentityIdentityCgradients/cond_grad/gradients/cond/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЃ
gradients/cond_grad/Identity_1IdentityEgradients/cond_grad/gradients/cond/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
gradients/cond_grad/Identity_2Identity=gradients/cond_grad/gradients/cond/split_grad/concat:output:0*
T0*
_output_shapes
:	@
gradients/cond_grad/Identity_3Identity?gradients/cond_grad/gradients/cond/split_1_grad/concat:output:0*
T0*
_output_shapes

:@
gradients/cond_grad/Identity_4Identity?gradients/cond_grad/gradients/cond/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:@Й
gradients/cond_grad/Identity_5IdentityVgradients/cond_grad/gradients/cond/CudnnRNNV3_grad/CudnnRNNBackpropV3:input_backprop:0*
T0*,
_output_shapes
:џџџџџџџџџ"E
gradients_cond_grad_identity%gradients/cond_grad/Identity:output:0"I
gradients_cond_grad_identity_1'gradients/cond_grad/Identity_1:output:0"I
gradients_cond_grad_identity_2'gradients/cond_grad/Identity_2:output:0"I
gradients_cond_grad_identity_3'gradients/cond_grad/Identity_3:output:0"I
gradients_cond_grad_identity_4'gradients/cond_grad/Identity_4:output:0"I
gradients_cond_grad_identity_5'gradients/cond_grad/Identity_5:output:0*(
_construction_contextkEagerRuntime*я
_input_shapesн
к:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : :џџџџџџџџџ: : : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : :\ X
'
_output_shapes
:џџџџџџџџџ
-
_user_specified_namegradients/grad_ys_0:`\
+
_output_shapes
:џџџџџџџџџ
-
_user_specified_namegradients/grad_ys_1:\X
'
_output_shapes
:џџџџџџџџџ
-
_user_specified_namegradients/grad_ys_2:\X
'
_output_shapes
:џџџџџџџџџ
-
_user_specified_namegradients/grad_ys_3:KG

_output_shapes
: 
-
_user_specified_namegradients/grad_ys_4:D@

_output_shapes
: 
&
_user_specified_namecond/Squeeze:GC

_output_shapes
: 
)
_user_specified_namecond/CudnnRNNV3:GC

_output_shapes
: 
)
_user_specified_namecond/CudnnRNNV3:GC

_output_shapes
: 
)
_user_specified_namecond/CudnnRNNV3:G	C

_output_shapes
: 
)
_user_specified_namecond/CudnnRNNV3:d
`
,
_output_shapes
:џџџџџџџџџ
0
_user_specified_namecond/CudnnRNNV3/inputs:GC

_output_shapes
: 
)
_user_specified_namecond/ExpandDims:IE

_output_shapes
: 
+
_user_specified_namecond/ExpandDims_1:EA

_output_shapes
: 
'
_user_specified_namecond/concat_1:@<

_output_shapes
: 
"
_user_specified_name
cond/Sum:GC

_output_shapes
: 
)
_user_specified_namecond/CudnnRNNV3:_[
'
_output_shapes
:џџџџџџџџџ
0
_user_specified_namecond/ExpandDims/init_h:a]
'
_output_shapes
:џџџџџџџџџ
2
_user_specified_namecond/ExpandDims_1/init_c:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: 
рK
Ђ
(__forward_gpu_lstm_with_fallback_1316833

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_splitW

zeros_likeConst*
_output_shapes
:@*
dtype0*
valueB@*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Ю
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_2d8db02e-7366-49a3-aeb4-eef0d5b6c5e4*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1316658_1316834*
go_backwards( *

time_major( :T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
Ї
Ѓ
*__inference_model_10_layer_call_fn_1316874
input_11
unknown:	@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_10_layer_call_and_return_conditional_losses_1316404o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_11:'#
!
_user_specified_name	1316858:'#
!
_user_specified_name	1316860:'#
!
_user_specified_name	1316862:'#
!
_user_specified_name	1316864:'#
!
_user_specified_name	1316866:'#
!
_user_specified_name	1316868:'#
!
_user_specified_name	1316870
фE
ц
cond_while_body_1313816&
"cond_while_cond_while_loop_counter,
(cond_while_cond_while_maximum_iterations
cond_while_placeholder
cond_while_placeholder_1
cond_while_placeholder_2
cond_while_placeholder_3
cond_while_placeholder_4#
cond_while_cond_strided_slice_0a
]cond_while_tensorarrayv2read_tensorlistgetitem_cond_tensorarrayunstack_tensorlistfromtensor_0e
acond_while_tensorarrayv2read_1_tensorlistgetitem_cond_tensorarrayunstack_1_tensorlistfromtensor_0
cond_while_matmul_kernel_0*
&cond_while_matmul_1_recurrent_kernel_0
cond_while_biasadd_bias_0
cond_while_identity
cond_while_identity_1
cond_while_identity_2
cond_while_identity_3
cond_while_identity_4
cond_while_identity_5
cond_while_identity_6!
cond_while_cond_strided_slice_
[cond_while_tensorarrayv2read_tensorlistgetitem_cond_tensorarrayunstack_tensorlistfromtensorc
_cond_while_tensorarrayv2read_1_tensorlistgetitem_cond_tensorarrayunstack_1_tensorlistfromtensor
cond_while_matmul_kernel(
$cond_while_matmul_1_recurrent_kernel
cond_while_biasadd_bias
<cond/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Р
.cond/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]cond_while_tensorarrayv2read_tensorlistgetitem_cond_tensorarrayunstack_tensorlistfromtensor_0cond_while_placeholderEcond/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
>cond/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ч
0cond/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemacond_while_tensorarrayv2read_1_tensorlistgetitem_cond_tensorarrayunstack_1_tensorlistfromtensor_0cond_while_placeholderGcond/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
 
cond/while/MatMulMatMul5cond/while/TensorArrayV2Read/TensorListGetItem:item:0cond_while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@
cond/while/MatMul_1MatMulcond_while_placeholder_3&cond_while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@
cond/while/addAddV2cond/while/MatMul:product:0cond/while/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@~
cond/while/BiasAddBiasAddcond/while/add:z:0cond_while_biasadd_bias_0*
T0*'
_output_shapes
:џџџџџџџџџ@\
cond/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :з
cond/while/splitSplit#cond/while/split/split_dim:output:0cond/while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitj
cond/while/SigmoidSigmoidcond/while/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџl
cond/while/Sigmoid_1Sigmoidcond/while/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ{
cond/while/mulMulcond/while/Sigmoid_1:y:0cond_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџd
cond/while/TanhTanhcond/while/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџv
cond/while/mul_1Mulcond/while/Sigmoid:y:0cond/while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџu
cond/while/add_1AddV2cond/while/mul:z:0cond/while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџl
cond/while/Sigmoid_2Sigmoidcond/while/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџa
cond/while/Tanh_1Tanhcond/while/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџz
cond/while/mul_2Mulcond/while/Sigmoid_2:y:0cond/while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџj
cond/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      І
cond/while/TileTile7cond/while/TensorArrayV2Read_1/TensorListGetItem:item:0"cond/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
cond/while/SelectV2SelectV2cond/while/Tile:output:0cond/while/mul_2:z:0cond_while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџl
cond/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      Њ
cond/while/Tile_1Tile7cond/while/TensorArrayV2Read_1/TensorListGetItem:item:0$cond/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџl
cond/while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      Њ
cond/while/Tile_2Tile7cond/while/TensorArrayV2Read_1/TensorListGetItem:item:0$cond/while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
cond/while/SelectV2_1SelectV2cond/while/Tile_1:output:0cond/while/mul_2:z:0cond_while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ
cond/while/SelectV2_2SelectV2cond/while/Tile_2:output:0cond/while/add_1:z:0cond_while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџw
5cond/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ќ
/cond/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemcond_while_placeholder_1>cond/while/TensorArrayV2Write/TensorListSetItem/index:output:0cond/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшвT
cond/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :o
cond/while/add_2AddV2cond_while_placeholdercond/while/add_2/y:output:0*
T0*
_output_shapes
: T
cond/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :{
cond/while/add_3AddV2"cond_while_cond_while_loop_countercond/while/add_3/y:output:0*
T0*
_output_shapes
: V
cond/while/IdentityIdentitycond/while/add_3:z:0*
T0*
_output_shapes
: l
cond/while/Identity_1Identity(cond_while_cond_while_maximum_iterations*
T0*
_output_shapes
: X
cond/while/Identity_2Identitycond/while/add_2:z:0*
T0*
_output_shapes
: 
cond/while/Identity_3Identity?cond/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: q
cond/while/Identity_4Identitycond/while/SelectV2:output:0*
T0*'
_output_shapes
:џџџџџџџџџs
cond/while/Identity_5Identitycond/while/SelectV2_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџs
cond/while/Identity_6Identitycond/while/SelectV2_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"4
cond_while_biasadd_biascond_while_biasadd_bias_0"@
cond_while_cond_strided_slicecond_while_cond_strided_slice_0"3
cond_while_identitycond/while/Identity:output:0"7
cond_while_identity_1cond/while/Identity_1:output:0"7
cond_while_identity_2cond/while/Identity_2:output:0"7
cond_while_identity_3cond/while/Identity_3:output:0"7
cond_while_identity_4cond/while/Identity_4:output:0"7
cond_while_identity_5cond/while/Identity_5:output:0"7
cond_while_identity_6cond/while/Identity_6:output:0"N
$cond_while_matmul_1_recurrent_kernel&cond_while_matmul_1_recurrent_kernel_0"6
cond_while_matmul_kernelcond_while_matmul_kernel_0"Ф
_cond_while_tensorarrayv2read_1_tensorlistgetitem_cond_tensorarrayunstack_1_tensorlistfromtensoracond_while_tensorarrayv2read_1_tensorlistgetitem_cond_tensorarrayunstack_1_tensorlistfromtensor_0"М
[cond_while_tensorarrayv2read_tensorlistgetitem_cond_tensorarrayunstack_tensorlistfromtensor]cond_while_tensorarrayv2read_tensorlistgetitem_cond_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b: : : : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : :	@:@:@:O K

_output_shapes
: 
1
_user_specified_namecond/while/loop_counter:UQ

_output_shapes
: 
7
_user_specified_namecond/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:JF

_output_shapes
: 
,
_user_specified_namecond/strided_slice:d`

_output_shapes
: 
F
_user_specified_name.,cond/TensorArrayUnstack/TensorListFromTensor:f	b

_output_shapes
: 
H
_user_specified_name0.cond/TensorArrayUnstack_1/TensorListFromTensor:G
C

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
г
Д
)__inference_lstm_10_layer_call_fn_1317007

inputs
unknown:	@
	unknown_0:@
	unknown_1:@
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lstm_10_layer_call_and_return_conditional_losses_1316363o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:'#
!
_user_specified_name	1316999:'#
!
_user_specified_name	1317001:'#
!
_user_specified_name	1317003
,
а
while_body_1318375
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@v
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@o
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:џџџџџџџџџ@W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ш
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
:	@
 
_user_specified_namekernel:P	L

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@
<

_output_shapes
:@

_user_specified_namebias
О
Ћ
cond_false_1313669_rewritten
cond_expanddims_mask

cond_matmul_1_init_h
cond_mul_init_c
cond_matmul_kernel"
cond_matmul_1_recurrent_kernel
cond_biasadd_bias
cond_transpose_inputs
cond_identity
cond_identity_1
cond_identity_2
cond_identity_3
cond_identity_4
cond_optionalfromvalue
cond_optionalfromvalue_1
cond_optionalfromvalue_2
cond_optionalfromvalue_3
cond_optionalfromvalue_4
cond_optionalfromvalue_5
cond_optionalfromvalue_6
cond_optionalfromvalue_7
cond_optionalfromvalue_8
cond_optionalfromvalue_9
cond_optionalfromvalue_10
cond_optionalfromvalue_11
cond_optionalfromvalue_12
cond_optionalfromvalue_13
cond_optionalfromvalue_14
cond_optionalfromvalue_15
cond_optionalfromvalue_16
cond_optionalfromvalue_17
cond_optionalfromvalue_18
cond_optionalfromvalue_19
cond_optionalfromvalue_20
cond_optionalfromvalue_21
cond_optionalfromvalue_22
cond_optionalfromvalue_23
cond_optionalfromvalue_24
cond_optionalfromvalue_25
cond_optionalfromvalue_26
cond_optionalfromvalue_27
cond_optionalfromvalue_28
cond_optionalfromvalue_29
cond_optionalfromvalue_30
cond_optionalfromvalue_31
cond_optionalfromvalue_32
cond_optionalfromvalue_33
cond_optionalfromvalue_34
cond_optionalfromvalue_35
cond_optionalfromvalue_36h
cond/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
cond/transpose	Transposecond_transpose_inputscond/transpose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџZ

cond/ShapeShapecond/transpose:y:0*
T0*
_output_shapes
::эЯb
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
cond/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
cond/ExpandDims
ExpandDimscond_expanddims_maskcond/ExpandDims/dim:output:0*
T0
*+
_output_shapes
:џџџџџџџџџj
cond/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
cond/transpose_1	Transposecond/ExpandDims:output:0cond/transpose_1/perm:output:0*
T0
*+
_output_shapes
:џџџџџџџџџk
 cond/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџС
cond/TensorArrayV2TensorListReserve)cond/TensorArrayV2/element_shape:output:0cond/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
:cond/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   я
,cond/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorcond/transpose:y:0Ccond/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвd
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
cond/strided_slice_1StridedSlicecond/transpose:y:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskz
cond/MatMulMatMulcond/strided_slice_1:output:0cond_matmul_kernel*
T0*'
_output_shapes
:џџџџџџџџџ@
cond/MatMul_1MatMulcond_matmul_1_init_hcond_matmul_1_recurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ@s
cond/addAddV2cond/MatMul:product:0cond/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@j
cond/BiasAddBiasAddcond/add:z:0cond_biasadd_bias*
T0*'
_output_shapes
:џџџџџџџџџ@V
cond/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х

cond/splitSplitcond/split/split_dim:output:0cond/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split^
cond/SigmoidSigmoidcond/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
cond/Sigmoid_1Sigmoidcond/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџf
cond/mulMulcond/Sigmoid_1:y:0cond_mul_init_c*
T0*'
_output_shapes
:џџџџџџџџџX
	cond/TanhTanhcond/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd

cond/mul_1Mulcond/Sigmoid:y:0cond/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџc

cond/add_1AddV2cond/mul:z:0cond/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ`
cond/Sigmoid_2Sigmoidcond/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџU
cond/Tanh_1Tanhcond/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџh

cond/mul_2Mulcond/Sigmoid_2:y:0cond/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџs
"cond/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   c
!cond/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :д
cond/TensorArrayV2_1TensorListReserve+cond/TensorArrayV2_1/element_shape:output:0*cond/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвK
	cond/timeConst*
_output_shapes
: *
dtype0*
value	B : m
"cond/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџХ
cond/TensorArrayV2_2TensorListReserve+cond/TensorArrayV2_2/element_shape:output:0cond/strided_slice:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
<cond/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ѕ
.cond/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorcond/transpose_1:y:0Econd/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ^
cond/zeros_like	ZerosLikecond/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџh
cond/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџY
cond/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ђ

cond/whileStatelessWhile cond/while/loop_counter:output:0&cond/while/maximum_iterations:output:0cond/time:output:0cond/TensorArrayV2_1:handle:0cond/zeros_like:y:0cond_matmul_1_init_hcond_mul_init_ccond/strided_slice:output:0<cond/TensorArrayUnstack/TensorListFromTensor:output_handle:0>cond/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0cond_matmul_kernelcond_matmul_1_recurrent_kernelcond_biasadd_bias/cond/cond/while/SelectV2_0/accumulator:handle:0-cond/cond/while/Tile_1_0/accumulator:handle:0cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_0/accumulator:handle:0cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_1_0/accumulator:handle:0cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_2_0/accumulator:handle:0cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_3_0/accumulator:handle:0-cond/cond/while/Tile_2_0/accumulator:handle:0cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_0/accumulator:handle:0cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_1_0/accumulator:handle:0cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_2_0/accumulator:handle:0cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_3_0/accumulator:handle:0+cond/cond/while/Tile_0/accumulator:handle:0}cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_0/accumulator:handle:0cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_2_0/accumulator:handle:0-cond/cond/while/Tanh_1_0/accumulator:handle:00cond/cond/while/Sigmoid_2_0/accumulator:handle:0zcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Shape_0/accumulator:handle:0|cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Shape_1_0/accumulator:handle:04cond/cond/while/Placeholder_4_0/accumulator:handle:00cond/cond/while/Sigmoid_1_0/accumulator:handle:0+cond/cond/while/Tanh_0/accumulator:handle:0.cond/cond/while/Sigmoid_0/accumulator:handle:0xcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Shape_0/accumulator:handle:0zcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Shape_1_0/accumulator:handle:0Jcond/cond/while/TensorArrayV2Read/TensorListGetItem_0/accumulator:handle:04cond/cond/while/Placeholder_3_0/accumulator:handle:0Љcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListElementShape_0/accumulator:handle:0Ѓcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListLength_0/accumulator:handle:02cond/cond/while/Placeholder_0/accumulator:handle:0*3
T.
,2**
_lower_using_switch_merge(*
_num_original_outputs**В
_output_shapes
: : : : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : :	@:@:@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : * 
_read_only_resource_inputs
 *-
body%R#
!cond_while_body_1313816_rewritten*-
cond%R#
!cond_while_cond_1313815_rewritten*Б
output_shapes
: : : : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : :	@:@:@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : *
parallel_iterations 
5cond/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   х
'cond/TensorArrayV2Stack/TensorListStackTensorListStackcond/while:output:3>cond/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsm
cond/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџf
cond/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
cond/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB: 
cond/strided_slice_2StridedSlice0cond/TensorArrayV2Stack/TensorListStack:tensor:0#cond/strided_slice_2/stack:output:0%cond/strided_slice_2/stack_1:output:0%cond/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskj
cond/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ѕ
cond/transpose_2	Transpose0cond/TensorArrayV2Stack/TensorListStack:tensor:0cond/transpose_2/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ`
cond/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?j
cond/IdentityIdentitycond/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџg
cond/Identity_1Identitycond/transpose_2:y:0*
T0*+
_output_shapes
:џџџџџџџџџb
cond/Identity_2Identitycond/while:output:5*
T0*'
_output_shapes
:џџџџџџџџџb
cond/Identity_3Identitycond/while:output:6*
T0*'
_output_shapes
:џџџџџџџџџS
cond/Identity_4Identitycond/runtime:output:0*
T0*
_output_shapes
: Є
cond/OptionalFromValueOptionalFromValue0cond/TensorArrayV2Stack/TensorListStack:tensor:0*
Toutput_types
2*
_output_shapes
: :ъшв
cond/OptionalFromValue_1OptionalFromValuecond/while:output:4*
Toutput_types
2*
_output_shapes
: :ъшв
cond/OptionalFromValue_2OptionalFromValuecond/while:output:8*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_3OptionalFromValuecond/while:output:10*
Toutput_types
2*
_output_shapes
: :ъшв
cond/OptionalFromValue_4OptionalFromValuecond/while:output:11*
Toutput_types
2*
_output_shapes
: :ъшв
cond/OptionalFromValue_5OptionalFromValuecond/while:output:12*
Toutput_types
2*
_output_shapes
: :ъшв
4cond/cond/while/SelectV2_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ђ
&cond/cond/while/SelectV2_0/accumulatorEmptyTensorList=cond/cond/while/SelectV2_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
2cond/cond/while/Tile_1_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ю
$cond/cond/while/Tile_1_0/accumulatorEmptyTensorList;cond/cond/while/Tile_1_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШЯ
cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
vcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_0/accumulatorEmptyTensorListcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшЯб
cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_1_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
xcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_1_0/accumulatorEmptyTensorListcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_1_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшЯб
cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_2_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
xcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_2_0/accumulatorEmptyTensorListcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_2_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшЯб
cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_3_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
xcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_3_0/accumulatorEmptyTensorListcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_3_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшЯ
2cond/cond/while/Tile_2_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ю
$cond/cond/while/Tile_2_0/accumulatorEmptyTensorList;cond/cond/while/Tile_2_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШЯ
cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
vcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_0/accumulatorEmptyTensorListcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшЯб
cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_1_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
xcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_1_0/accumulatorEmptyTensorListcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_1_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшЯб
cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_2_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
xcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_2_0/accumulatorEmptyTensorListcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_2_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшЯб
cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_3_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
xcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_3_0/accumulatorEmptyTensorListcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_3_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшЯ
0cond/cond/while/Tile_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ъ
"cond/cond/while/Tile_0/accumulatorEmptyTensorList9cond/cond/while/Tile_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШЭ
cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
tcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_0/accumulatorEmptyTensorListcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшЯЯ
cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_2_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
vcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_2_0/accumulatorEmptyTensorListcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_2_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшЯ
2cond/cond/while/Tanh_1_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ю
$cond/cond/while/Tanh_1_0/accumulatorEmptyTensorList;cond/cond/while/Tanh_1_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5cond/cond/while/Sigmoid_2_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   є
'cond/cond/while/Sigmoid_2_0/accumulatorEmptyTensorList>cond/cond/while/Sigmoid_2_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвЩ
cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Shape_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
qcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Shape_0/accumulatorEmptyTensorListcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Shape_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшЯЬ
cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Shape_1_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
scond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Shape_1_0/accumulatorEmptyTensorListcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Shape_1_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшЯ
9cond/cond/while/Placeholder_4_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ќ
+cond/cond/while/Placeholder_4_0/accumulatorEmptyTensorListBcond/cond/while/Placeholder_4_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5cond/cond/while/Sigmoid_1_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   є
'cond/cond/while/Sigmoid_1_0/accumulatorEmptyTensorList>cond/cond/while/Sigmoid_1_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
0cond/cond/while/Tanh_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ъ
"cond/cond/while/Tanh_0/accumulatorEmptyTensorList9cond/cond/while/Tanh_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
3cond/cond/while/Sigmoid_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   №
%cond/cond/while/Sigmoid_0/accumulatorEmptyTensorList<cond/cond/while/Sigmoid_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвЧ
}cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Shape_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
ocond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Shape_0/accumulatorEmptyTensorListcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Shape_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшЯЩ
cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Shape_1_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
qcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Shape_1_0/accumulatorEmptyTensorListcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Shape_1_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшЯ 
Ocond/cond/while/TensorArrayV2Read/TensorListGetItem_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ј
Acond/cond/while/TensorArrayV2Read/TensorListGetItem_0/accumulatorEmptyTensorListXcond/cond/while/TensorArrayV2Read/TensorListGetItem_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
9cond/cond/while/Placeholder_3_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ќ
+cond/cond/while/Placeholder_3_0/accumulatorEmptyTensorListBcond/cond/while/Placeholder_3_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвљ
Ўcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListElementShape_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:ш
 cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListElementShape_0/accumulatorEmptyTensorListЗcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListElementShape_0/accumulator/element_shape:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшЯM

cond/ConstConst*
_output_shapes
: *
dtype0*
valueB Н
cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListLength_0/accumulatorEmptyTensorListcond/Const:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшЯO
cond/Const_1Const*
_output_shapes
: *
dtype0*
valueB Э
)cond/cond/while/Placeholder_0/accumulatorEmptyTensorListcond/Const_1:output:0&cond/while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшЯ
cond/OptionalFromValue_6OptionalFromValuecond/while:output:0*
Toutput_types
2*
_output_shapes
: :ъшЯ
cond/OptionalFromValue_7OptionalFromValuecond/while:output:13*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_8OptionalFromValuecond/while:output:14*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_9OptionalFromValuecond/while:output:15*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_10OptionalFromValuecond/while:output:16*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_11OptionalFromValuecond/while:output:17*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_12OptionalFromValuecond/while:output:18*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_13OptionalFromValuecond/while:output:19*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_14OptionalFromValuecond/while:output:20*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_15OptionalFromValuecond/while:output:21*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_16OptionalFromValuecond/while:output:22*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_17OptionalFromValuecond/while:output:23*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_18OptionalFromValuecond/while:output:24*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_19OptionalFromValuecond/while:output:25*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_20OptionalFromValuecond/while:output:26*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_21OptionalFromValuecond/while:output:27*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_22OptionalFromValuecond/while:output:28*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_23OptionalFromValuecond/while:output:29*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_24OptionalFromValuecond/while:output:30*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_25OptionalFromValuecond/while:output:31*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_26OptionalFromValuecond/while:output:32*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_27OptionalFromValuecond/while:output:33*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_28OptionalFromValuecond/while:output:34*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_29OptionalFromValuecond/while:output:35*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_30OptionalFromValuecond/while:output:36*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_31OptionalFromValuecond/while:output:37*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_32OptionalFromValuecond/while:output:38*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_33OptionalFromValuecond/while:output:39*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_34OptionalFromValuecond/while:output:40*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_35OptionalFromValuecond/while:output:41*
Toutput_types
2*
_output_shapes
: :ъшлO
cond/OptionalFromValue_36OptionalFromValuecond/transpose:y:0*
Toutput_types
2*
_output_shapes
: :ъшв"'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0"+
cond_identity_2cond/Identity_2:output:0"+
cond_identity_3cond/Identity_3:output:0"+
cond_identity_4cond/Identity_4:output:0";
cond_optionalfromvalue!cond/OptionalFromValue:optional:0"?
cond_optionalfromvalue_1#cond/OptionalFromValue_1:optional:0"A
cond_optionalfromvalue_10$cond/OptionalFromValue_10:optional:0"A
cond_optionalfromvalue_11$cond/OptionalFromValue_11:optional:0"A
cond_optionalfromvalue_12$cond/OptionalFromValue_12:optional:0"A
cond_optionalfromvalue_13$cond/OptionalFromValue_13:optional:0"A
cond_optionalfromvalue_14$cond/OptionalFromValue_14:optional:0"A
cond_optionalfromvalue_15$cond/OptionalFromValue_15:optional:0"A
cond_optionalfromvalue_16$cond/OptionalFromValue_16:optional:0"A
cond_optionalfromvalue_17$cond/OptionalFromValue_17:optional:0"A
cond_optionalfromvalue_18$cond/OptionalFromValue_18:optional:0"A
cond_optionalfromvalue_19$cond/OptionalFromValue_19:optional:0"?
cond_optionalfromvalue_2#cond/OptionalFromValue_2:optional:0"A
cond_optionalfromvalue_20$cond/OptionalFromValue_20:optional:0"A
cond_optionalfromvalue_21$cond/OptionalFromValue_21:optional:0"A
cond_optionalfromvalue_22$cond/OptionalFromValue_22:optional:0"A
cond_optionalfromvalue_23$cond/OptionalFromValue_23:optional:0"A
cond_optionalfromvalue_24$cond/OptionalFromValue_24:optional:0"A
cond_optionalfromvalue_25$cond/OptionalFromValue_25:optional:0"A
cond_optionalfromvalue_26$cond/OptionalFromValue_26:optional:0"A
cond_optionalfromvalue_27$cond/OptionalFromValue_27:optional:0"A
cond_optionalfromvalue_28$cond/OptionalFromValue_28:optional:0"A
cond_optionalfromvalue_29$cond/OptionalFromValue_29:optional:0"?
cond_optionalfromvalue_3#cond/OptionalFromValue_3:optional:0"A
cond_optionalfromvalue_30$cond/OptionalFromValue_30:optional:0"A
cond_optionalfromvalue_31$cond/OptionalFromValue_31:optional:0"A
cond_optionalfromvalue_32$cond/OptionalFromValue_32:optional:0"A
cond_optionalfromvalue_33$cond/OptionalFromValue_33:optional:0"A
cond_optionalfromvalue_34$cond/OptionalFromValue_34:optional:0"A
cond_optionalfromvalue_35$cond/OptionalFromValue_35:optional:0"A
cond_optionalfromvalue_36$cond/OptionalFromValue_36:optional:0"?
cond_optionalfromvalue_4#cond/OptionalFromValue_4:optional:0"?
cond_optionalfromvalue_5#cond/OptionalFromValue_5:optional:0"?
cond_optionalfromvalue_6#cond/OptionalFromValue_6:optional:0"?
cond_optionalfromvalue_7#cond/OptionalFromValue_7:optional:0"?
cond_optionalfromvalue_8#cond/OptionalFromValue_8:optional:0"?
cond_optionalfromvalue_9#cond/OptionalFromValue_9:optional:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@:џџџџџџџџџ:M I
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias:TP
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
к&
Д
.cond_while_cond_1313815_rewritten_grad_1314632=
9gradients_cond_grad_gradients_cond_while_grad_placeholder?
;gradients_cond_grad_gradients_cond_while_grad_placeholder_1?
;gradients_cond_grad_gradients_cond_while_grad_placeholder_2?
;gradients_cond_grad_gradients_cond_while_grad_placeholder_3?
;gradients_cond_grad_gradients_cond_while_grad_placeholder_4?
;gradients_cond_grad_gradients_cond_while_grad_placeholder_5?
;gradients_cond_grad_gradients_cond_while_grad_placeholder_6?
;gradients_cond_grad_gradients_cond_while_grad_placeholder_7?
;gradients_cond_grad_gradients_cond_while_grad_placeholder_8?
;gradients_cond_grad_gradients_cond_while_grad_placeholder_9@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_10@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_11@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_12@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_13@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_14@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_15@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_16@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_17@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_18@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_19@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_20@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_21@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_22@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_23@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_24@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_25@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_26@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_27@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_28@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_29@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_30@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_31@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_32@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_33@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_34@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_35@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_36@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_37@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_38@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_39@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_40@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_41:
6gradients_cond_grad_gradients_cond_while_grad_identity
г
2gradients/cond_grad/gradients/cond/while_grad/LessLess9gradients_cond_grad_gradients_cond_while_grad_placeholder;gradients_cond_grad_gradients_cond_while_grad_placeholder_2*
T0*
_output_shapes
: 
6gradients/cond_grad/gradients/cond/while_grad/IdentityIdentity6gradients/cond_grad/gradients/cond/while_grad/Less:z:0*
T0
*
_output_shapes
: "y
6gradients_cond_grad_gradients_cond_while_grad_identity?gradients/cond_grad/gradients/cond/while_grad/Identity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
­: : : : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: :	@:@:@: : : : : : : : : : : : : : : : : : : : : : : : :	@: :@: : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :%!

_output_shapes
:	@:$	 

_output_shapes

:@: 


_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :%#!

_output_shapes
:	@:$

_output_shapes
: :$% 

_output_shapes

:@:&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: 
Ї
Ѓ
*__inference_model_10_layer_call_fn_1316893
input_11
unknown:	@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*)
_read_only_resource_inputs
	*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_model_10_layer_call_and_return_conditional_losses_1316855o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_11:'#
!
_user_specified_name	1316877:'#
!
_user_specified_name	1316879:'#
!
_user_specified_name	1316881:'#
!
_user_specified_name	1316883:'#
!
_user_specified_name	1316885:'#
!
_user_specified_name	1316887:'#
!
_user_specified_name	1316889
Э)
о
*__inference_gpu_lstm_with_fallback_1313935

inputs

init_h

init_c

kernel
recurrent_kernel
bias
mask

identity

identity_1

identity_2

identity_3

identity_4ЂcondG
ShapeShapemask*
T0
*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskS
CastCastmask*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџW
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :b
SumSumCast:y:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:џџџџџџџџџT
SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : V
SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :
SequenceMask/RangeRangeSequenceMask/Const:output:0strided_slice:output:0SequenceMask/Const_1:output:0*
_output_shapes
:f
SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
SequenceMask/ExpandDims
ExpandDimsSum:output:0$SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџ|
SequenceMask/CastCast SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ
SequenceMask/LessLessSequenceMask/Range:output:0SequenceMask/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ]
EqualEqualmaskSequenceMask/Less:z:0*
T0
*'
_output_shapes
:џџџџџџџџџV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       =
AllAll	Equal:z:0Const:output:0*
_output_shapes
: G

LogicalNot
LogicalNotmask*'
_output_shapes
:џџџџџџџџџY
All_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :c
All_1AllLogicalNot:y:0 All_1/reduction_indices:output:0*#
_output_shapes
:џџџџџџџџџQ
Const_1Const*
_output_shapes
:*
dtype0*
valueB: D
AnyAnyAll_1:output:0Const_1:output:0*
_output_shapes
: @
LogicalNot_1
LogicalNotAny:output:0*
_output_shapes
: P

LogicalAnd
LogicalAndAll:output:0LogicalNot_1:y:0*
_output_shapes
: И
condIfLogicalAnd:z:0maskinit_hinit_ckernelrecurrent_kernelbiasinputs*
Tcond0
*
Tin
	2
*
Tout	
2*
_lower_using_switch_merge(*В
_output_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : * 
_read_only_resource_inputs
 *%
else_branchR
cond_false_1313669*e
output_shapesT
R:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: *$
then_branchR
cond_true_1313668Z
cond/IdentityIdentitycond:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
cond/Identity_1Identitycond:output:1*
T0*+
_output_shapes
:џџџџџџџџџ\
cond/Identity_2Identitycond:output:2*
T0*'
_output_shapes
:џџџџџџџџџ\
cond/Identity_3Identitycond:output:3*
T0*'
_output_shapes
:џџџџџџџџџK
cond/Identity_4Identitycond:output:4*
T0*
_output_shapes
: e
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџm

Identity_1Identitycond/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџi

Identity_2Identitycond/Identity_2:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџi

Identity_3Identitycond/Identity_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџX

Identity_4Identitycond/Identity_4:output:0^NoOp*
T0*
_output_shapes
: )
NoOpNoOp^cond*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@:џџџџџџџџџ*=
api_implements+)lstm_4b11e03b-047e-44a3-bf0d-d903c89a5c70*
api_preferred_deviceGPU*
go_backwards( *

time_major( 2
condcond:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias:MI
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask
§
М
D__inference_lstm_10_layer_call_and_return_conditional_losses_1316836

inputs/
read_readvariableop_resource:	@0
read_1_readvariableop_resource:@,
read_2_readvariableop_resource:@

identity_3ЂRead/ReadVariableOpЂRead_1/ReadVariableOpЂRead_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	@*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	@t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:@*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:@*
dtype0Z

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@Ж
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference_standard_lstm_1316563i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџh
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
з

while_cond_1313524
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_less_strided_slice5
1while_while_cond_1313524___redundant_placeholder05
1while_while_cond_1313524___redundant_placeholder15
1while_while_cond_1313524___redundant_placeholder25
1while_while_cond_1313524___redundant_placeholder35
1while_while_cond_1313524___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: ::::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:

_output_shapes
::	

_output_shapes
::


_output_shapes
::

_output_shapes
::

_output_shapes
:
ї>
З
while_body_1313525
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0[
Wwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorY
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
9while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ў
+while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_placeholderBwhile/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0

while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@
while/MatMul_1MatMulwhile_placeholder_3!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@v
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@o
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:џџџџџџџџџ@W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ш
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџe
while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      

while/TileTile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
while/SelectV2SelectV2while/Tile:output:0while/mul_2:z:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџg
while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      
while/Tile_1Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџg
while/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      
while/Tile_2Tile2while/TensorArrayV2Read_1/TensorListGetItem:item:0while/Tile_2/multiples:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ
while/SelectV2_1SelectV2while/Tile_1:output:0while/mul_2:z:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџ
while/SelectV2_2SelectV2while/Tile_2:output:0while/add_1:z:0while_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ш
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: g
while/Identity_4Identitywhile/SelectV2:output:0*
T0*'
_output_shapes
:џџџџџџџџџi
while/Identity_5Identitywhile/SelectV2_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџi
while/Identity_6Identitywhile/SelectV2_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"А
Uwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_tensorarrayv2read_1_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b: : : : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : :	@:@:@:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:a	]

_output_shapes
: 
C
_user_specified_name+)TensorArrayUnstack_1/TensorListFromTensor:G
C

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
Й
P
$__inference__update_step_xla_1285116
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:@: *
	_noinline(:H D

_output_shapes

:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ћK
Ђ
(__forward_gpu_lstm_with_fallback_1315466

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_splitW

zeros_likeConst*
_output_shapes
:@*
dtype0*
valueB@*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0з
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_40d6fd70-48b3-4841-ad66-b73fbb417689*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1315291_1315467*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias

П
#__inference__traced_restore_1319084
file_prefix2
 assignvariableop_dense_20_kernel:.
 assignvariableop_1_dense_20_bias:4
"assignvariableop_2_dense_21_kernel:.
 assignvariableop_3_dense_21_bias:>
+assignvariableop_4_lstm_10_lstm_cell_kernel:	@G
5assignvariableop_5_lstm_10_lstm_cell_recurrent_kernel:@7
)assignvariableop_6_lstm_10_lstm_cell_bias:@&
assignvariableop_7_iteration:	 2
(assignvariableop_8_current_learning_rate: E
2assignvariableop_9_adam_m_lstm_10_lstm_cell_kernel:	@F
3assignvariableop_10_adam_v_lstm_10_lstm_cell_kernel:	@O
=assignvariableop_11_adam_m_lstm_10_lstm_cell_recurrent_kernel:@O
=assignvariableop_12_adam_v_lstm_10_lstm_cell_recurrent_kernel:@?
1assignvariableop_13_adam_m_lstm_10_lstm_cell_bias:@?
1assignvariableop_14_adam_v_lstm_10_lstm_cell_bias:@<
*assignvariableop_15_adam_m_dense_20_kernel:<
*assignvariableop_16_adam_v_dense_20_kernel:6
(assignvariableop_17_adam_m_dense_20_bias:6
(assignvariableop_18_adam_v_dense_20_bias:<
*assignvariableop_19_adam_m_dense_21_kernel:<
*assignvariableop_20_adam_v_dense_21_kernel:6
(assignvariableop_21_adam_m_dense_21_bias:6
(assignvariableop_22_adam_v_dense_21_bias:%
assignvariableop_23_total_1: %
assignvariableop_24_count_1: #
assignvariableop_25_total: #
assignvariableop_26_count: 2
$assignvariableop_27_true_positives_1:1
#assignvariableop_28_false_positives:0
"assignvariableop_29_true_positives:1
#assignvariableop_30_false_negatives:
identity_32ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9ѕ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_current_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHА
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B С
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOpAssignVariableOp assignvariableop_dense_20_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_20_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_21_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_21_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_4AssignVariableOp+assignvariableop_4_lstm_10_lstm_cell_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_5AssignVariableOp5assignvariableop_5_lstm_10_lstm_cell_recurrent_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_6AssignVariableOp)assignvariableop_6_lstm_10_lstm_cell_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:Г
AssignVariableOp_7AssignVariableOpassignvariableop_7_iterationIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_8AssignVariableOp(assignvariableop_8_current_learning_rateIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_9AssignVariableOp2assignvariableop_9_adam_m_lstm_10_lstm_cell_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_10AssignVariableOp3assignvariableop_10_adam_v_lstm_10_lstm_cell_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_11AssignVariableOp=assignvariableop_11_adam_m_lstm_10_lstm_cell_recurrent_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_12AssignVariableOp=assignvariableop_12_adam_v_lstm_10_lstm_cell_recurrent_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_13AssignVariableOp1assignvariableop_13_adam_m_lstm_10_lstm_cell_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_14AssignVariableOp1assignvariableop_14_adam_v_lstm_10_lstm_cell_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_m_dense_20_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_v_dense_20_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_m_dense_20_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_v_dense_20_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_m_dense_21_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_v_dense_21_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_m_dense_21_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_v_dense_21_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_27AssignVariableOp$assignvariableop_27_true_positives_1Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_28AssignVariableOp#assignvariableop_28_false_positivesIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_29AssignVariableOp"assignvariableop_29_true_positivesIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_30AssignVariableOp#assignvariableop_30_false_negativesIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 љ
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: Т
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_32Identity_32:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:/+
)
_user_specified_namedense_20/kernel:-)
'
_user_specified_namedense_20/bias:/+
)
_user_specified_namedense_21/kernel:-)
'
_user_specified_namedense_21/bias:84
2
_user_specified_namelstm_10/lstm_cell/kernel:B>
<
_user_specified_name$"lstm_10/lstm_cell/recurrent_kernel:62
0
_user_specified_namelstm_10/lstm_cell/bias:)%
#
_user_specified_name	iteration:5	1
/
_user_specified_namecurrent_learning_rate:?
;
9
_user_specified_name!Adam/m/lstm_10/lstm_cell/kernel:?;
9
_user_specified_name!Adam/v/lstm_10/lstm_cell/kernel:IE
C
_user_specified_name+)Adam/m/lstm_10/lstm_cell/recurrent_kernel:IE
C
_user_specified_name+)Adam/v/lstm_10/lstm_cell/recurrent_kernel:=9
7
_user_specified_nameAdam/m/lstm_10/lstm_cell/bias:=9
7
_user_specified_nameAdam/v/lstm_10/lstm_cell/bias:62
0
_user_specified_nameAdam/m/dense_20/kernel:62
0
_user_specified_nameAdam/v/dense_20/kernel:40
.
_user_specified_nameAdam/m/dense_20/bias:40
.
_user_specified_nameAdam/v/dense_20/bias:62
0
_user_specified_nameAdam/m/dense_21/kernel:62
0
_user_specified_nameAdam/v/dense_21/kernel:40
.
_user_specified_nameAdam/m/dense_21/bias:40
.
_user_specified_nameAdam/v/dense_21/bias:'#
!
_user_specified_name	total_1:'#
!
_user_specified_name	count_1:%!

_user_specified_nametotal:%!

_user_specified_namecount:0,
*
_user_specified_nametrue_positives_1:/+
)
_user_specified_namefalse_positives:.*
(
_user_specified_nametrue_positives:/+
)
_user_specified_namefalse_negatives
­
L
$__inference__update_step_xla_1285141
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

О
D__inference_lstm_10_layer_call_and_return_conditional_losses_1317876
inputs_0/
read_readvariableop_resource:	@0
read_1_readvariableop_resource:@,
read_2_readvariableop_resource:@

identity_3ЂRead/ReadVariableOpЂRead_1/ReadVariableOpЂRead_2/ReadVariableOpK
ShapeShapeinputs_0*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	@*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	@t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:@*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:@*
dtype0Z

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@И
PartitionedCallPartitionedCallinputs_0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference_standard_lstm_1317603i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџh
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџ: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_0:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ъ
ч
=__inference___backward_gpu_lstm_with_fallback_1318127_1318303
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
::эЯЊ
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯЄ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯЈ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџХ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯ
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Є
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*b
_output_shapesP
N:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:б
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:џџџџџџџџџ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
::эЯХ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
::эЯЩ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:ј
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ђ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ѓ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ѓ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Ж
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:З
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:З
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Џ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	@Д
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:@\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: e
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@g
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:@Ъ
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::б
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:@е
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:@s
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:џџџџџџџџџt

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	@g

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:@c

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:@"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*ї
_input_shapesх
т:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::џџџџџџџџџ:џџџџџџџџџ: ::::::::: : : : *=
api_implements+)lstm_771db2a2-7bde-4d3a-9c4a-c316a68cc663*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1318302*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ:1-
+
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_namestrided_slice:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:B	>

_output_shapes
:
"
_user_specified_name
CudnnRNN:W
S
,
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	transpose:WS
+
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
ExpandDims:YU
+
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameExpandDims_1:FB

_output_shapes

:
"
_user_specified_name
concat_1:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:EA

_output_shapes
: 
'
_user_specified_nameconcat_1/axis:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_8/perm:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:C?

_output_shapes
: 
%
_user_specified_nameconcat/axis
Њ@
Э
*__inference_gpu_lstm_with_fallback_1316184

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_splitW

zeros_likeConst*
_output_shapes
:@*
dtype0*
valueB@*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:Ъ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_0eacfb0a-82c4-46c0-8e97-a277e1646ed4*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
,
а
while_body_1317517
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@v
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@o
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:џџџџџџџџџ@W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ш
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
:	@
 
_user_specified_namekernel:P	L

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@
<

_output_shapes
:@

_user_specified_namebias
рK
Ђ
(__forward_gpu_lstm_with_fallback_1318731

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_splitW

zeros_likeConst*
_output_shapes
:@*
dtype0*
valueB@*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Ю
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_7f07c2bf-ca9b-4883-99d3-8a7b5b1d9c08*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1318556_1318732*
go_backwards( *

time_major( :T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
ж?
Д
"__inference__wrapped_model_1315040
input_11@
-model_10_lstm_10_read_readvariableop_resource:	@A
/model_10_lstm_10_read_1_readvariableop_resource:@=
/model_10_lstm_10_read_2_readvariableop_resource:@B
0model_10_dense_20_matmul_readvariableop_resource:?
1model_10_dense_20_biasadd_readvariableop_resource:B
0model_10_dense_21_matmul_readvariableop_resource:?
1model_10_dense_21_biasadd_readvariableop_resource:
identityЂ(model_10/dense_20/BiasAdd/ReadVariableOpЂ'model_10/dense_20/MatMul/ReadVariableOpЂ(model_10/dense_21/BiasAdd/ReadVariableOpЂ'model_10/dense_21/MatMul/ReadVariableOpЂ$model_10/lstm_10/Read/ReadVariableOpЂ&model_10/lstm_10/Read_1/ReadVariableOpЂ&model_10/lstm_10/Read_2/ReadVariableOpc
model_10/masking_10/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
model_10/masking_10/NotEqualNotEqualinput_11'model_10/masking_10/NotEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџt
)model_10/masking_10/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
model_10/masking_10/AnyAny model_10/masking_10/NotEqual:z:02model_10/masking_10/Any/reduction_indices:output:0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(
model_10/masking_10/CastCast model_10/masking_10/Any:output:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџ}
model_10/masking_10/mulMulinput_11model_10/masking_10/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџ
model_10/masking_10/SqueezeSqueeze model_10/masking_10/Any:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims

џџџџџџџџџo
model_10/lstm_10/ShapeShapemodel_10/masking_10/mul:z:0*
T0*
_output_shapes
::эЯn
$model_10/lstm_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&model_10/lstm_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&model_10/lstm_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:І
model_10/lstm_10/strided_sliceStridedSlicemodel_10/lstm_10/Shape:output:0-model_10/lstm_10/strided_slice/stack:output:0/model_10/lstm_10/strided_slice/stack_1:output:0/model_10/lstm_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model_10/lstm_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :І
model_10/lstm_10/zeros/packedPack'model_10/lstm_10/strided_slice:output:0(model_10/lstm_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:a
model_10/lstm_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
model_10/lstm_10/zerosFill&model_10/lstm_10/zeros/packed:output:0%model_10/lstm_10/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
!model_10/lstm_10/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Њ
model_10/lstm_10/zeros_1/packedPack'model_10/lstm_10/strided_slice:output:0*model_10/lstm_10/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:c
model_10/lstm_10/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ѕ
model_10/lstm_10/zeros_1Fill(model_10/lstm_10/zeros_1/packed:output:0'model_10/lstm_10/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
$model_10/lstm_10/Read/ReadVariableOpReadVariableOp-model_10_lstm_10_read_readvariableop_resource*
_output_shapes
:	@*
dtype0}
model_10/lstm_10/IdentityIdentity,model_10/lstm_10/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	@
&model_10/lstm_10/Read_1/ReadVariableOpReadVariableOp/model_10_lstm_10_read_1_readvariableop_resource*
_output_shapes

:@*
dtype0
model_10/lstm_10/Identity_1Identity.model_10/lstm_10/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@
&model_10/lstm_10/Read_2/ReadVariableOpReadVariableOp/model_10_lstm_10_read_2_readvariableop_resource*
_output_shapes
:@*
dtype0|
model_10/lstm_10/Identity_2Identity.model_10/lstm_10/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@и
 model_10/lstm_10/PartitionedCallPartitionedCallmodel_10/masking_10/mul:z:0model_10/lstm_10/zeros:output:0!model_10/lstm_10/zeros_1:output:0"model_10/lstm_10/Identity:output:0$model_10/lstm_10/Identity_1:output:0$model_10/lstm_10/Identity_2:output:0$model_10/masking_10/Squeeze:output:0*
Tin
	2
*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference_standard_lstm_1313629
'model_10/dense_20/MatMul/ReadVariableOpReadVariableOp0model_10_dense_20_matmul_readvariableop_resource*
_output_shapes

:*
dtype0А
model_10/dense_20/MatMulMatMul)model_10/lstm_10/PartitionedCall:output:0/model_10/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
(model_10/dense_20/BiasAdd/ReadVariableOpReadVariableOp1model_10_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
model_10/dense_20/BiasAddBiasAdd"model_10/dense_20/MatMul:product:00model_10/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџt
model_10/dense_20/ReluRelu"model_10/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
'model_10/dense_21/MatMul/ReadVariableOpReadVariableOp0model_10_dense_21_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ћ
model_10/dense_21/MatMulMatMul$model_10/dense_20/Relu:activations:0/model_10/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
(model_10/dense_21/BiasAdd/ReadVariableOpReadVariableOp1model_10_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
model_10/dense_21/BiasAddBiasAdd"model_10/dense_21/MatMul:product:00model_10/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџz
model_10/dense_21/SigmoidSigmoid"model_10/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџl
IdentityIdentitymodel_10/dense_21/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџХ
NoOpNoOp)^model_10/dense_20/BiasAdd/ReadVariableOp(^model_10/dense_20/MatMul/ReadVariableOp)^model_10/dense_21/BiasAdd/ReadVariableOp(^model_10/dense_21/MatMul/ReadVariableOp%^model_10/lstm_10/Read/ReadVariableOp'^model_10/lstm_10/Read_1/ReadVariableOp'^model_10/lstm_10/Read_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ: : : : : : : 2T
(model_10/dense_20/BiasAdd/ReadVariableOp(model_10/dense_20/BiasAdd/ReadVariableOp2R
'model_10/dense_20/MatMul/ReadVariableOp'model_10/dense_20/MatMul/ReadVariableOp2T
(model_10/dense_21/BiasAdd/ReadVariableOp(model_10/dense_21/BiasAdd/ReadVariableOp2R
'model_10/dense_21/MatMul/ReadVariableOp'model_10/dense_21/MatMul/ReadVariableOp2L
$model_10/lstm_10/Read/ReadVariableOp$model_10/lstm_10/Read/ReadVariableOp2P
&model_10/lstm_10/Read_1/ReadVariableOp&model_10/lstm_10/Read_1/ReadVariableOp2P
&model_10/lstm_10/Read_2/ReadVariableOp&model_10/lstm_10/Read_2/ReadVariableOp:V R
,
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
input_11:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ю@
Э
*__inference_gpu_lstm_with_fallback_1317697

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_splitW

zeros_likeConst*
_output_shapes
:@*
dtype0*
valueB@*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:г
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_47cdb55f-aa18-46f4-a131-7cc1fe3544f6*
api_preferred_deviceGPU*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
Њ@
Э
*__inference_gpu_lstm_with_fallback_1318126

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_splitW

zeros_likeConst*
_output_shapes
:@*
dtype0*
valueB@*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:Ъ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_771db2a2-7bde-4d3a-9c4a-c316a68cc663*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
Њ;
С
!__inference_standard_lstm_1315196

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџP
ShapeShapetranspose:y:0*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ@^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ@d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:џџџџџџџџџ@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@* 
_read_only_resource_inputs
 *
bodyR
while_body_1315110*
condR
while_cond_1315109*`
output_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_40d6fd70-48b3-4841-ad66-b73fbb417689*
api_preferred_deviceCPU*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
кЪ
ч
=__inference___backward_gpu_lstm_with_fallback_1315720_1315896
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
::эЯЊ
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯЄ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯЈ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџХ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯ
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:Ђ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:­
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*k
_output_shapesY
W:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:к
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
::эЯХ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
::эЯЩ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:ј
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ђ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ѓ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ѓ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Ж
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:З
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:З
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Џ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	@Д
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:@\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: e
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@g
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:@Ъ
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::б
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:@е
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:@|
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџt

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	@g

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:@c

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:@"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapesї
є:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ::џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::џџџџџџџџџ:џџџџџџџџџ: ::::::::: : : : *=
api_implements+)lstm_ded2654b-8cb5-4033-bb74-72cb92ca8019*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1315895*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ:1-
+
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_namestrided_slice:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:^Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
CudnnRNN:B	>

_output_shapes
:
"
_user_specified_name
CudnnRNN:`
\
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	transpose:WS
+
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
ExpandDims:YU
+
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameExpandDims_1:FB

_output_shapes

:
"
_user_specified_name
concat_1:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:EA

_output_shapes
: 
'
_user_specified_nameconcat_1/axis:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_8/perm:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:C?

_output_shapes
: 
%
_user_specified_nameconcat/axis
Ц
c
G__inference_masking_10_layer_call_and_return_conditional_losses_1315933

inputs
identityO

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    h
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџv
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(_
CastCastAny:output:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџS
mulMulinputsCast:y:0*
T0*,
_output_shapes
:џџџџџџџџџr
SqueezeSqueezeAny:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims

џџџџџџџџџT
IdentityIdentitymul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


cond_while_cond_1313815&
"cond_while_cond_while_loop_counter,
(cond_while_cond_while_maximum_iterations
cond_while_placeholder
cond_while_placeholder_1
cond_while_placeholder_2
cond_while_placeholder_3
cond_while_placeholder_4&
"cond_while_less_cond_strided_slice?
;cond_while_cond_while_cond_1313815___redundant_placeholder0?
;cond_while_cond_while_cond_1313815___redundant_placeholder1?
;cond_while_cond_while_cond_1313815___redundant_placeholder2?
;cond_while_cond_while_cond_1313815___redundant_placeholder3?
;cond_while_cond_while_cond_1313815___redundant_placeholder4
cond_while_identity
t
cond/while/LessLesscond_while_placeholder"cond_while_less_cond_strided_slice*
T0*
_output_shapes
: U
cond/while/IdentityIdentitycond/while/Less:z:0*
T0
*
_output_shapes
: "3
cond_while_identitycond/while/Identity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W: : : : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: ::::::O K

_output_shapes
: 
1
_user_specified_namecond/while/loop_counter:UQ

_output_shapes
: 
7
_user_specified_namecond/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:JF

_output_shapes
: 
,
_user_specified_namecond/strided_slice:

_output_shapes
::	

_output_shapes
::


_output_shapes
::

_output_shapes
::

_output_shapes
:
Ъ
ч
=__inference___backward_gpu_lstm_with_fallback_1316185_1316361
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
::эЯЊ
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯЄ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯЈ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџХ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯ
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Є
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*b
_output_shapesP
N:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:б
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:џџџџџџџџџ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
::эЯХ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
::эЯЩ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:ј
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ђ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ѓ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ѓ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Ж
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:З
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:З
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Џ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	@Д
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:@\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: e
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@g
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:@Ъ
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::б
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:@е
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:@s
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:џџџџџџџџџt

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	@g

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:@c

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:@"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*ї
_input_shapesх
т:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::џџџџџџџџџ:џџџџџџџџџ: ::::::::: : : : *=
api_implements+)lstm_0eacfb0a-82c4-46c0-8e97-a277e1646ed4*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1316360*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ:1-
+
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_namestrided_slice:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:B	>

_output_shapes
:
"
_user_specified_name
CudnnRNN:W
S
,
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	transpose:WS
+
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
ExpandDims:YU
+
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameExpandDims_1:FB

_output_shapes

:
"
_user_specified_name
concat_1:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:EA

_output_shapes
: 
'
_user_specified_nameconcat_1/axis:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_8/perm:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:C?

_output_shapes
: 
%
_user_specified_nameconcat/axis
F
Ы
!__inference_standard_lstm_1313629

inputs

init_h

init_c

kernel
recurrent_kernel
bias
mask

identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџP
ShapeShapetranspose:y:0*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџm

ExpandDims
ExpandDimsmaskExpandDims/dim:output:0*
T0
*+
_output_shapes
:џџџџџџџџџe
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ~
transpose_1	TransposeExpandDims:output:0transpose_1/perm:output:0*
T0
*+
_output_shapes
:џџџџџџџџџf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ@^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ@d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:џџџџџџџџџ@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : h
TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЖ
TensorArrayV2_2TensorListReserve&TensorArrayV2_2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ц
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_1:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШT

zeros_like	ZerosLike	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџc
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros_like:y:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*v
_output_shapesd
b: : : : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : :	@:@:@* 
_read_only_resource_inputs
 *
bodyR
while_body_1313525*
condR
while_cond_1313524*u
output_shapesd
b: : : : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : :	@:@:@*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_2	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_2/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ]

Identity_1Identitytranspose_2:y:0*
T0*+
_output_shapes
:џџџџџџџџџX

Identity_2Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџX

Identity_3Identitywhile:output:6*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@:џџџџџџџџџ*=
api_implements+)lstm_4b11e03b-047e-44a3-bf0d-d903c89a5c70*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias:MI
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask
Ь

і
E__inference_dense_20_layer_call_and_return_conditional_losses_1318754

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
яE
Ч
!cond_while_cond_1313815_rewritten&
"cond_while_cond_while_loop_counter,
(cond_while_cond_while_maximum_iterations
cond_while_placeholder
cond_while_placeholder_1
cond_while_placeholder_2
cond_while_placeholder_3
cond_while_placeholder_4&
"cond_while_less_cond_strided_slice?
;cond_while_cond_while_cond_1313815___redundant_placeholder0?
;cond_while_cond_while_cond_1313815___redundant_placeholder1?
;cond_while_cond_while_cond_1313815___redundant_placeholder2?
;cond_while_cond_while_cond_1313815___redundant_placeholder3?
;cond_while_cond_while_cond_1313815___redundant_placeholder45
1cond_while_cond_cond_while_selectv2_0_accumulator3
/cond_while_cond_cond_while_tile_1_0_accumulator
cond_while_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shape_0_accumulator
cond_while_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shape_1_0_accumulator
cond_while_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shape_2_0_accumulator
cond_while_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shape_3_0_accumulator3
/cond_while_cond_cond_while_tile_2_0_accumulator
cond_while_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shape_0_accumulator
cond_while_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shape_1_0_accumulator
cond_while_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shape_2_0_accumulator
cond_while_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shape_3_0_accumulator1
-cond_while_cond_cond_while_tile_0_accumulator
cond_while_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_shape_0_accumulator
cond_while_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_shape_2_0_accumulator3
/cond_while_cond_cond_while_tanh_1_0_accumulator6
2cond_while_cond_cond_while_sigmoid_2_0_accumulator
|cond_while_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_1_grad_shape_0_accumulator
~cond_while_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_1_grad_shape_1_0_accumulator:
6cond_while_cond_cond_while_placeholder_4_0_accumulator6
2cond_while_cond_cond_while_sigmoid_1_0_accumulator1
-cond_while_cond_cond_while_tanh_0_accumulator4
0cond_while_cond_cond_while_sigmoid_0_accumulator~
zcond_while_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_grad_shape_0_accumulator
|cond_while_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_grad_shape_1_0_accumulatorP
Lcond_while_cond_cond_while_tensorarrayv2read_tensorlistgetitem_0_accumulator:
6cond_while_cond_cond_while_placeholder_3_0_accumulatorА
Ћcond_while_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistelementshape_0_accumulatorЊ
Ѕcond_while_cond_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistlength_0_accumulator8
4cond_while_cond_cond_while_placeholder_0_accumulator
cond_while_identity
t
cond/while/LessLesscond_while_placeholder"cond_while_less_cond_strided_slice*
T0*
_output_shapes
: U
cond/while/IdentityIdentitycond/while/Less:z:0*
T0
*
_output_shapes
: "3
cond_while_identitycond/while/Identity:output:0*(
_construction_contextkEagerRuntime*І
_input_shapes
: : : : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: :::::: : : : : : : : : : : : : : : : : : : : : : : : : : : : : :O K

_output_shapes
: 
1
_user_specified_namecond/while/loop_counter:UQ

_output_shapes
: 
7
_user_specified_namecond/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:JF

_output_shapes
: 
,
_user_specified_namecond/strided_slice:

_output_shapes
::	

_output_shapes
::


_output_shapes
::

_output_shapes
::

_output_shapes
::^Z

_output_shapes
: 
@
_user_specified_name(&cond/cond/while/SelectV2_0/accumulator:\X

_output_shapes
: 
>
_user_specified_name&$cond/cond/while/Tile_1_0/accumulator:АЋ

_output_shapes
: 

_user_specified_namexvcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_0/accumulator:В­

_output_shapes
: 

_user_specified_namezxcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_1_0/accumulator:В­

_output_shapes
: 

_user_specified_namezxcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_2_0/accumulator:В­

_output_shapes
: 

_user_specified_namezxcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_3_0/accumulator:\X

_output_shapes
: 
>
_user_specified_name&$cond/cond/while/Tile_2_0/accumulator:АЋ

_output_shapes
: 

_user_specified_namexvcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_0/accumulator:В­

_output_shapes
: 

_user_specified_namezxcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_1_0/accumulator:В­

_output_shapes
: 

_user_specified_namezxcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_2_0/accumulator:В­

_output_shapes
: 

_user_specified_namezxcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_3_0/accumulator:ZV

_output_shapes
: 
<
_user_specified_name$"cond/cond/while/Tile_0/accumulator:ЎЉ

_output_shapes
: 

_user_specified_namevtcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_0/accumulator:АЋ

_output_shapes
: 

_user_specified_namexvcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_2_0/accumulator:\X

_output_shapes
: 
>
_user_specified_name&$cond/cond/while/Tanh_1_0/accumulator:_[

_output_shapes
: 
A
_user_specified_name)'cond/cond/while/Sigmoid_2_0/accumulator:ЋІ

_output_shapes
: 

_user_specified_namesqcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Shape_0/accumulator:­Ј

_output_shapes
: 

_user_specified_nameuscond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Shape_1_0/accumulator:c_

_output_shapes
: 
E
_user_specified_name-+cond/cond/while/Placeholder_4_0/accumulator:_ [

_output_shapes
: 
A
_user_specified_name)'cond/cond/while/Sigmoid_1_0/accumulator:Z!V

_output_shapes
: 
<
_user_specified_name$"cond/cond/while/Tanh_0/accumulator:]"Y

_output_shapes
: 
?
_user_specified_name'%cond/cond/while/Sigmoid_0/accumulator:Љ#Є

_output_shapes
: 

_user_specified_nameqocond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Shape_0/accumulator:Ћ$І

_output_shapes
: 

_user_specified_namesqcond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Shape_1_0/accumulator:y%u

_output_shapes
: 
[
_user_specified_nameCAcond/cond/while/TensorArrayV2Read/TensorListGetItem_0/accumulator:c&_

_output_shapes
: 
E
_user_specified_name-+cond/cond/while/Placeholder_3_0/accumulator:м'з

_output_shapes
: 
М
_user_specified_nameЃ cond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListElementShape_0/accumulator:ж(б

_output_shapes
: 
Ж
_user_specified_namecond/cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListLength_0/accumulator:a)]

_output_shapes
: 
C
_user_specified_name+)cond/cond/while/Placeholder_0/accumulator
Њ;
С
!__inference_standard_lstm_1317603

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџP
ShapeShapetranspose:y:0*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ@^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ@d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:џџџџџџџџџ@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@* 
_read_only_resource_inputs
 *
bodyR
while_body_1317517*
condR
while_cond_1317516*`
output_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_47cdb55f-aa18-46f4-a131-7cc1fe3544f6*
api_preferred_deviceCPU*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
щy


cond_true_1313668_rewritten
cond_cast_mask

cond_expanddims_init_h
cond_expanddims_1_init_c
cond_split_kernel!
cond_split_1_recurrent_kernel
cond_concat_bias
cond_cudnnrnnv3_inputs
cond_identity
cond_identity_1
cond_identity_2
cond_identity_3
cond_identity_4
cond_optionalfromvalue
cond_optionalfromvalue_1
cond_optionalfromvalue_2
cond_optionalfromvalue_3
cond_optionalfromvalue_4
cond_optionalfromvalue_5
cond_optionalfromvalue_6
cond_optionalfromvalue_7
cond_optionalfromvalue_8
cond_optionalfromvalue_9
cond_optionalnone
cond_optionalnone_1
cond_optionalnone_2
cond_optionalnone_3
cond_optionalnone_4
cond_optionalnone_5
cond_optionalnone_6
cond_optionalnone_7
cond_optionalnone_8
cond_optionalnone_9
cond_optionalnone_10
cond_optionalnone_11
cond_optionalnone_12
cond_optionalnone_13
cond_optionalnone_14
cond_optionalnone_15
cond_optionalnone_16
cond_optionalnone_17
cond_optionalnone_18
cond_optionalnone_19
cond_optionalnone_20
cond_optionalnone_21
cond_optionalnone_22
cond_optionalnone_23
cond_optionalnone_24
cond_optionalnone_25
cond_optionalnone_26b
	cond/CastCastcond_cast_mask*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ\
cond/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :q
cond/SumSumcond/Cast:y:0#cond/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:џџџџџџџџџU
cond/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
cond/ExpandDims
ExpandDimscond_expanddims_init_hcond/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџW
cond/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
cond/ExpandDims_1
ExpandDimscond_expanddims_1_init_ccond/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџV
cond/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ё

cond/splitSplitcond/split/split_dim:output:0cond_split_kernel*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitX
cond/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :­
cond/split_1Splitcond/split_1/split_dim:output:0cond_split_1_recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_split\
cond/zeros_likeConst*
_output_shapes
:@*
dtype0*
valueB@*    R
cond/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
cond/concatConcatV2cond/zeros_like:output:0cond_concat_biascond/concat/axis:output:0*
N*
T0*
_output_shapes	
:X
cond/split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ќ
cond/split_2Splitcond/split_2/split_dim:output:0cond/concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_split]

cond/ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџd
cond/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       x
cond/transpose	Transposecond/split:output:0cond/transpose/perm:output:0*
T0*
_output_shapes
:	f
cond/ReshapeReshapecond/transpose:y:0cond/Const:output:0*
T0*
_output_shapes	
:@f
cond/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       |
cond/transpose_1	Transposecond/split:output:1cond/transpose_1/perm:output:0*
T0*
_output_shapes
:	j
cond/Reshape_1Reshapecond/transpose_1:y:0cond/Const:output:0*
T0*
_output_shapes	
:@f
cond/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       |
cond/transpose_2	Transposecond/split:output:2cond/transpose_2/perm:output:0*
T0*
_output_shapes
:	j
cond/Reshape_2Reshapecond/transpose_2:y:0cond/Const:output:0*
T0*
_output_shapes	
:@f
cond/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       |
cond/transpose_3	Transposecond/split:output:3cond/transpose_3/perm:output:0*
T0*
_output_shapes
:	j
cond/Reshape_3Reshapecond/transpose_3:y:0cond/Const:output:0*
T0*
_output_shapes	
:@f
cond/transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       }
cond/transpose_4	Transposecond/split_1:output:0cond/transpose_4/perm:output:0*
T0*
_output_shapes

:j
cond/Reshape_4Reshapecond/transpose_4:y:0cond/Const:output:0*
T0*
_output_shapes	
:f
cond/transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       }
cond/transpose_5	Transposecond/split_1:output:1cond/transpose_5/perm:output:0*
T0*
_output_shapes

:j
cond/Reshape_5Reshapecond/transpose_5:y:0cond/Const:output:0*
T0*
_output_shapes	
:f
cond/transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       }
cond/transpose_6	Transposecond/split_1:output:2cond/transpose_6/perm:output:0*
T0*
_output_shapes

:j
cond/Reshape_6Reshapecond/transpose_6:y:0cond/Const:output:0*
T0*
_output_shapes	
:f
cond/transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       }
cond/transpose_7	Transposecond/split_1:output:3cond/transpose_7/perm:output:0*
T0*
_output_shapes

:j
cond/Reshape_7Reshapecond/transpose_7:y:0cond/Const:output:0*
T0*
_output_shapes	
:j
cond/Reshape_8Reshapecond/split_2:output:0cond/Const:output:0*
T0*
_output_shapes
:j
cond/Reshape_9Reshapecond/split_2:output:1cond/Const:output:0*
T0*
_output_shapes
:k
cond/Reshape_10Reshapecond/split_2:output:2cond/Const:output:0*
T0*
_output_shapes
:k
cond/Reshape_11Reshapecond/split_2:output:3cond/Const:output:0*
T0*
_output_shapes
:k
cond/Reshape_12Reshapecond/split_2:output:4cond/Const:output:0*
T0*
_output_shapes
:k
cond/Reshape_13Reshapecond/split_2:output:5cond/Const:output:0*
T0*
_output_shapes
:k
cond/Reshape_14Reshapecond/split_2:output:6cond/Const:output:0*
T0*
_output_shapes
:k
cond/Reshape_15Reshapecond/split_2:output:7cond/Const:output:0*
T0*
_output_shapes
:T
cond/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
cond/concat_1ConcatV2cond/Reshape:output:0cond/Reshape_1:output:0cond/Reshape_2:output:0cond/Reshape_3:output:0cond/Reshape_4:output:0cond/Reshape_5:output:0cond/Reshape_6:output:0cond/Reshape_7:output:0cond/Reshape_8:output:0cond/Reshape_9:output:0cond/Reshape_10:output:0cond/Reshape_11:output:0cond/Reshape_12:output:0cond/Reshape_13:output:0cond/Reshape_14:output:0cond/Reshape_15:output:0cond/concat_1/axis:output:0*
N*
T0*
_output_shapes

:
cond/CudnnRNNV3
CudnnRNNV3cond_cudnnrnnv3_inputscond/ExpandDims:output:0cond/ExpandDims_1:output:0cond/concat_1:output:0cond/Sum:output:0*
T0*a
_output_shapesO
M:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::*

time_major( k
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџd
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ї
cond/strided_sliceStridedSlicecond/CudnnRNNV3:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask|
cond/SqueezeSqueezecond/CudnnRNNV3:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
~
cond/Squeeze_1Squeezecond/CudnnRNNV3:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
W
cond/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
cond/ExpandDims_2
ExpandDimscond/Squeeze:output:0cond/ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ`
cond/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @b
cond/IdentityIdentitycond/Squeeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџm
cond/Identity_1Identitycond/ExpandDims_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџd
cond/Identity_2Identitycond/Squeeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
cond/Identity_3Identitycond/Squeeze_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџS
cond/Identity_4Identitycond/runtime:output:0*
T0*
_output_shapes
: 
cond/OptionalFromValueOptionalFromValuecond/Squeeze:output:0*
Toutput_types
2*
_output_shapes
: :ъшв
cond/OptionalFromValue_1OptionalFromValuecond/CudnnRNNV3:output_c:0*
Toutput_types
2*
_output_shapes
: :ъшв
cond/OptionalFromValue_2OptionalFromValuecond/CudnnRNNV3:output_h:0*
Toutput_types
2*
_output_shapes
: :ъшв
cond/OptionalFromValue_3OptionalFromValuecond/CudnnRNNV3:output:0*
Toutput_types
2*
_output_shapes
: :ъшв
cond/OptionalFromValue_4OptionalFromValuecond/CudnnRNNV3:reserve_space:0*
Toutput_types
2*
_output_shapes
: :ъшв
cond/OptionalFromValue_5OptionalFromValuecond/ExpandDims:output:0*
Toutput_types
2*
_output_shapes
: :ъшв
cond/OptionalFromValue_6OptionalFromValuecond/ExpandDims_1:output:0*
Toutput_types
2*
_output_shapes
: :ъшв
cond/OptionalFromValue_7OptionalFromValuecond/concat_1:output:0*
Toutput_types
2*
_output_shapes
: :ъшв
cond/OptionalFromValue_8OptionalFromValuecond/Sum:output:0*
Toutput_types
2*
_output_shapes
: :ъшЯ
cond/OptionalFromValue_9OptionalFromValuecond/CudnnRNNV3:host_reserved:0*
Toutput_types
2*
_output_shapes
: :ъшЭ9
cond/OptionalNoneOptionalNone*
_output_shapes
: ;
cond/OptionalNone_1OptionalNone*
_output_shapes
: ;
cond/OptionalNone_2OptionalNone*
_output_shapes
: ;
cond/OptionalNone_3OptionalNone*
_output_shapes
: ;
cond/OptionalNone_4OptionalNone*
_output_shapes
: ;
cond/OptionalNone_5OptionalNone*
_output_shapes
: ;
cond/OptionalNone_6OptionalNone*
_output_shapes
: ;
cond/OptionalNone_7OptionalNone*
_output_shapes
: ;
cond/OptionalNone_8OptionalNone*
_output_shapes
: ;
cond/OptionalNone_9OptionalNone*
_output_shapes
: <
cond/OptionalNone_10OptionalNone*
_output_shapes
: <
cond/OptionalNone_11OptionalNone*
_output_shapes
: <
cond/OptionalNone_12OptionalNone*
_output_shapes
: <
cond/OptionalNone_13OptionalNone*
_output_shapes
: <
cond/OptionalNone_14OptionalNone*
_output_shapes
: <
cond/OptionalNone_15OptionalNone*
_output_shapes
: <
cond/OptionalNone_16OptionalNone*
_output_shapes
: <
cond/OptionalNone_17OptionalNone*
_output_shapes
: <
cond/OptionalNone_18OptionalNone*
_output_shapes
: <
cond/OptionalNone_19OptionalNone*
_output_shapes
: <
cond/OptionalNone_20OptionalNone*
_output_shapes
: <
cond/OptionalNone_21OptionalNone*
_output_shapes
: <
cond/OptionalNone_22OptionalNone*
_output_shapes
: <
cond/OptionalNone_23OptionalNone*
_output_shapes
: <
cond/OptionalNone_24OptionalNone*
_output_shapes
: <
cond/OptionalNone_25OptionalNone*
_output_shapes
: <
cond/OptionalNone_26OptionalNone*
_output_shapes
: "'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0"+
cond_identity_2cond/Identity_2:output:0"+
cond_identity_3cond/Identity_3:output:0"+
cond_identity_4cond/Identity_4:output:0";
cond_optionalfromvalue!cond/OptionalFromValue:optional:0"?
cond_optionalfromvalue_1#cond/OptionalFromValue_1:optional:0"?
cond_optionalfromvalue_2#cond/OptionalFromValue_2:optional:0"?
cond_optionalfromvalue_3#cond/OptionalFromValue_3:optional:0"?
cond_optionalfromvalue_4#cond/OptionalFromValue_4:optional:0"?
cond_optionalfromvalue_5#cond/OptionalFromValue_5:optional:0"?
cond_optionalfromvalue_6#cond/OptionalFromValue_6:optional:0"?
cond_optionalfromvalue_7#cond/OptionalFromValue_7:optional:0"?
cond_optionalfromvalue_8#cond/OptionalFromValue_8:optional:0"?
cond_optionalfromvalue_9#cond/OptionalFromValue_9:optional:0"1
cond_optionalnonecond/OptionalNone:optional:0"5
cond_optionalnone_1cond/OptionalNone_1:optional:0"7
cond_optionalnone_10cond/OptionalNone_10:optional:0"7
cond_optionalnone_11cond/OptionalNone_11:optional:0"7
cond_optionalnone_12cond/OptionalNone_12:optional:0"7
cond_optionalnone_13cond/OptionalNone_13:optional:0"7
cond_optionalnone_14cond/OptionalNone_14:optional:0"7
cond_optionalnone_15cond/OptionalNone_15:optional:0"7
cond_optionalnone_16cond/OptionalNone_16:optional:0"7
cond_optionalnone_17cond/OptionalNone_17:optional:0"7
cond_optionalnone_18cond/OptionalNone_18:optional:0"7
cond_optionalnone_19cond/OptionalNone_19:optional:0"5
cond_optionalnone_2cond/OptionalNone_2:optional:0"7
cond_optionalnone_20cond/OptionalNone_20:optional:0"7
cond_optionalnone_21cond/OptionalNone_21:optional:0"7
cond_optionalnone_22cond/OptionalNone_22:optional:0"7
cond_optionalnone_23cond/OptionalNone_23:optional:0"7
cond_optionalnone_24cond/OptionalNone_24:optional:0"7
cond_optionalnone_25cond/OptionalNone_25:optional:0"7
cond_optionalnone_26cond/OptionalNone_26:optional:0"5
cond_optionalnone_3cond/OptionalNone_3:optional:0"5
cond_optionalnone_4cond/OptionalNone_4:optional:0"5
cond_optionalnone_5cond/OptionalNone_5:optional:0"5
cond_optionalnone_6cond/OptionalNone_6:optional:0"5
cond_optionalnone_7cond/OptionalNone_7:optional:0"5
cond_optionalnone_8cond/OptionalNone_8:optional:0"5
cond_optionalnone_9cond/OptionalNone_9:optional:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@:џџџџџџџџџ:M I
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias:TP
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љё
в
 __inference__traced_save_1318982
file_prefix8
&read_disablecopyonread_dense_20_kernel:4
&read_1_disablecopyonread_dense_20_bias::
(read_2_disablecopyonread_dense_21_kernel:4
&read_3_disablecopyonread_dense_21_bias:D
1read_4_disablecopyonread_lstm_10_lstm_cell_kernel:	@M
;read_5_disablecopyonread_lstm_10_lstm_cell_recurrent_kernel:@=
/read_6_disablecopyonread_lstm_10_lstm_cell_bias:@,
"read_7_disablecopyonread_iteration:	 8
.read_8_disablecopyonread_current_learning_rate: K
8read_9_disablecopyonread_adam_m_lstm_10_lstm_cell_kernel:	@L
9read_10_disablecopyonread_adam_v_lstm_10_lstm_cell_kernel:	@U
Cread_11_disablecopyonread_adam_m_lstm_10_lstm_cell_recurrent_kernel:@U
Cread_12_disablecopyonread_adam_v_lstm_10_lstm_cell_recurrent_kernel:@E
7read_13_disablecopyonread_adam_m_lstm_10_lstm_cell_bias:@E
7read_14_disablecopyonread_adam_v_lstm_10_lstm_cell_bias:@B
0read_15_disablecopyonread_adam_m_dense_20_kernel:B
0read_16_disablecopyonread_adam_v_dense_20_kernel:<
.read_17_disablecopyonread_adam_m_dense_20_bias:<
.read_18_disablecopyonread_adam_v_dense_20_bias:B
0read_19_disablecopyonread_adam_m_dense_21_kernel:B
0read_20_disablecopyonread_adam_v_dense_21_kernel:<
.read_21_disablecopyonread_adam_m_dense_21_bias:<
.read_22_disablecopyonread_adam_v_dense_21_bias:+
!read_23_disablecopyonread_total_1: +
!read_24_disablecopyonread_count_1: )
read_25_disablecopyonread_total: )
read_26_disablecopyonread_count: 8
*read_27_disablecopyonread_true_positives_1:7
)read_28_disablecopyonread_false_positives:6
(read_29_disablecopyonread_true_positives:7
)read_30_disablecopyonread_false_negatives:
savev2_const
identity_63ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_18/DisableCopyOnReadЂRead_18/ReadVariableOpЂRead_19/DisableCopyOnReadЂRead_19/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_20/DisableCopyOnReadЂRead_20/ReadVariableOpЂRead_21/DisableCopyOnReadЂRead_21/ReadVariableOpЂRead_22/DisableCopyOnReadЂRead_22/ReadVariableOpЂRead_23/DisableCopyOnReadЂRead_23/ReadVariableOpЂRead_24/DisableCopyOnReadЂRead_24/ReadVariableOpЂRead_25/DisableCopyOnReadЂRead_25/ReadVariableOpЂRead_26/DisableCopyOnReadЂRead_26/ReadVariableOpЂRead_27/DisableCopyOnReadЂRead_27/ReadVariableOpЂRead_28/DisableCopyOnReadЂRead_28/ReadVariableOpЂRead_29/DisableCopyOnReadЂRead_29/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_30/DisableCopyOnReadЂRead_30/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_20_kernel"/device:CPU:0*
_output_shapes
 Ђ
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_20_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_20_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_20_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_dense_21_kernel"/device:CPU:0*
_output_shapes
 Ј
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_dense_21_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_dense_21_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_dense_21_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_4/DisableCopyOnReadDisableCopyOnRead1read_4_disablecopyonread_lstm_10_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 В
Read_4/ReadVariableOpReadVariableOp1read_4_disablecopyonread_lstm_10_lstm_cell_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	@
Read_5/DisableCopyOnReadDisableCopyOnRead;read_5_disablecopyonread_lstm_10_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 Л
Read_5/ReadVariableOpReadVariableOp;read_5_disablecopyonread_lstm_10_lstm_cell_recurrent_kernel^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0n
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_6/DisableCopyOnReadDisableCopyOnRead/read_6_disablecopyonread_lstm_10_lstm_cell_bias"/device:CPU:0*
_output_shapes
 Ћ
Read_6/ReadVariableOpReadVariableOp/read_6_disablecopyonread_lstm_10_lstm_cell_bias^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:@v
Read_7/DisableCopyOnReadDisableCopyOnRead"read_7_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_7/ReadVariableOpReadVariableOp"read_7_disablecopyonread_iteration^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0	*
_output_shapes
: 
Read_8/DisableCopyOnReadDisableCopyOnRead.read_8_disablecopyonread_current_learning_rate"/device:CPU:0*
_output_shapes
 І
Read_8/ReadVariableOpReadVariableOp.read_8_disablecopyonread_current_learning_rate^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_9/DisableCopyOnReadDisableCopyOnRead8read_9_disablecopyonread_adam_m_lstm_10_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 Й
Read_9/ReadVariableOpReadVariableOp8read_9_disablecopyonread_adam_m_lstm_10_lstm_cell_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@*
dtype0o
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@f
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:	@
Read_10/DisableCopyOnReadDisableCopyOnRead9read_10_disablecopyonread_adam_v_lstm_10_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 М
Read_10/ReadVariableOpReadVariableOp9read_10_disablecopyonread_adam_v_lstm_10_lstm_cell_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@*
dtype0p
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@f
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	@
Read_11/DisableCopyOnReadDisableCopyOnReadCread_11_disablecopyonread_adam_m_lstm_10_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 Х
Read_11/ReadVariableOpReadVariableOpCread_11_disablecopyonread_adam_m_lstm_10_lstm_cell_recurrent_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_12/DisableCopyOnReadDisableCopyOnReadCread_12_disablecopyonread_adam_v_lstm_10_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 Х
Read_12/ReadVariableOpReadVariableOpCread_12_disablecopyonread_adam_v_lstm_10_lstm_cell_recurrent_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_13/DisableCopyOnReadDisableCopyOnRead7read_13_disablecopyonread_adam_m_lstm_10_lstm_cell_bias"/device:CPU:0*
_output_shapes
 Е
Read_13/ReadVariableOpReadVariableOp7read_13_disablecopyonread_adam_m_lstm_10_lstm_cell_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_14/DisableCopyOnReadDisableCopyOnRead7read_14_disablecopyonread_adam_v_lstm_10_lstm_cell_bias"/device:CPU:0*
_output_shapes
 Е
Read_14/ReadVariableOpReadVariableOp7read_14_disablecopyonread_adam_v_lstm_10_lstm_cell_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_15/DisableCopyOnReadDisableCopyOnRead0read_15_disablecopyonread_adam_m_dense_20_kernel"/device:CPU:0*
_output_shapes
 В
Read_15/ReadVariableOpReadVariableOp0read_15_disablecopyonread_adam_m_dense_20_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_16/DisableCopyOnReadDisableCopyOnRead0read_16_disablecopyonread_adam_v_dense_20_kernel"/device:CPU:0*
_output_shapes
 В
Read_16/ReadVariableOpReadVariableOp0read_16_disablecopyonread_adam_v_dense_20_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_17/DisableCopyOnReadDisableCopyOnRead.read_17_disablecopyonread_adam_m_dense_20_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_17/ReadVariableOpReadVariableOp.read_17_disablecopyonread_adam_m_dense_20_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_18/DisableCopyOnReadDisableCopyOnRead.read_18_disablecopyonread_adam_v_dense_20_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_18/ReadVariableOpReadVariableOp.read_18_disablecopyonread_adam_v_dense_20_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_19/DisableCopyOnReadDisableCopyOnRead0read_19_disablecopyonread_adam_m_dense_21_kernel"/device:CPU:0*
_output_shapes
 В
Read_19/ReadVariableOpReadVariableOp0read_19_disablecopyonread_adam_m_dense_21_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_20/DisableCopyOnReadDisableCopyOnRead0read_20_disablecopyonread_adam_v_dense_21_kernel"/device:CPU:0*
_output_shapes
 В
Read_20/ReadVariableOpReadVariableOp0read_20_disablecopyonread_adam_v_dense_21_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_21/DisableCopyOnReadDisableCopyOnRead.read_21_disablecopyonread_adam_m_dense_21_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_21/ReadVariableOpReadVariableOp.read_21_disablecopyonread_adam_m_dense_21_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_22/DisableCopyOnReadDisableCopyOnRead.read_22_disablecopyonread_adam_v_dense_21_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_22/ReadVariableOpReadVariableOp.read_22_disablecopyonread_adam_v_dense_21_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_23/DisableCopyOnReadDisableCopyOnRead!read_23_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 
Read_23/ReadVariableOpReadVariableOp!read_23_disablecopyonread_total_1^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_24/DisableCopyOnReadDisableCopyOnRead!read_24_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 
Read_24/ReadVariableOpReadVariableOp!read_24_disablecopyonread_count_1^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_25/DisableCopyOnReadDisableCopyOnReadread_25_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_25/ReadVariableOpReadVariableOpread_25_disablecopyonread_total^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_26/DisableCopyOnReadDisableCopyOnReadread_26_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_26/ReadVariableOpReadVariableOpread_26_disablecopyonread_count^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_27/DisableCopyOnReadDisableCopyOnRead*read_27_disablecopyonread_true_positives_1"/device:CPU:0*
_output_shapes
 Ј
Read_27/ReadVariableOpReadVariableOp*read_27_disablecopyonread_true_positives_1^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_28/DisableCopyOnReadDisableCopyOnRead)read_28_disablecopyonread_false_positives"/device:CPU:0*
_output_shapes
 Ї
Read_28/ReadVariableOpReadVariableOp)read_28_disablecopyonread_false_positives^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_29/DisableCopyOnReadDisableCopyOnRead(read_29_disablecopyonread_true_positives"/device:CPU:0*
_output_shapes
 І
Read_29/ReadVariableOpReadVariableOp(read_29_disablecopyonread_true_positives^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_30/DisableCopyOnReadDisableCopyOnRead)read_30_disablecopyonread_false_negatives"/device:CPU:0*
_output_shapes
 Ї
Read_30/ReadVariableOpReadVariableOp)read_30_disablecopyonread_false_negatives^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:ђ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_current_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH­
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *.
dtypes$
"2 	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_62Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_63IdentityIdentity_62:output:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_63Identity_63:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:/+
)
_user_specified_namedense_20/kernel:-)
'
_user_specified_namedense_20/bias:/+
)
_user_specified_namedense_21/kernel:-)
'
_user_specified_namedense_21/bias:84
2
_user_specified_namelstm_10/lstm_cell/kernel:B>
<
_user_specified_name$"lstm_10/lstm_cell/recurrent_kernel:62
0
_user_specified_namelstm_10/lstm_cell/bias:)%
#
_user_specified_name	iteration:5	1
/
_user_specified_namecurrent_learning_rate:?
;
9
_user_specified_name!Adam/m/lstm_10/lstm_cell/kernel:?;
9
_user_specified_name!Adam/v/lstm_10/lstm_cell/kernel:IE
C
_user_specified_name+)Adam/m/lstm_10/lstm_cell/recurrent_kernel:IE
C
_user_specified_name+)Adam/v/lstm_10/lstm_cell/recurrent_kernel:=9
7
_user_specified_nameAdam/m/lstm_10/lstm_cell/bias:=9
7
_user_specified_nameAdam/v/lstm_10/lstm_cell/bias:62
0
_user_specified_nameAdam/m/dense_20/kernel:62
0
_user_specified_nameAdam/v/dense_20/kernel:40
.
_user_specified_nameAdam/m/dense_20/bias:40
.
_user_specified_nameAdam/v/dense_20/bias:62
0
_user_specified_nameAdam/m/dense_21/kernel:62
0
_user_specified_nameAdam/v/dense_21/kernel:40
.
_user_specified_nameAdam/m/dense_21/bias:40
.
_user_specified_nameAdam/v/dense_21/bias:'#
!
_user_specified_name	total_1:'#
!
_user_specified_name	count_1:%!

_user_specified_nametotal:%!

_user_specified_namecount:0,
*
_user_specified_nametrue_positives_1:/+
)
_user_specified_namefalse_positives:.*
(
_user_specified_nametrue_positives:/+
)
_user_specified_namefalse_negatives:= 9

_output_shapes
: 

_user_specified_nameConst
Ю@
Э
*__inference_gpu_lstm_with_fallback_1315290

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_splitW

zeros_likeConst*
_output_shapes
:@*
dtype0*
valueB@*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:г
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_40d6fd70-48b3-4841-ad66-b73fbb417689*
api_preferred_deviceGPU*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
Т
МR
$cond_while_body_1313816_grad_1314184=
9gradients_cond_grad_gradients_cond_while_grad_placeholder?
;gradients_cond_grad_gradients_cond_while_grad_placeholder_1?
;gradients_cond_grad_gradients_cond_while_grad_placeholder_2?
;gradients_cond_grad_gradients_cond_while_grad_placeholder_3?
;gradients_cond_grad_gradients_cond_while_grad_placeholder_4?
;gradients_cond_grad_gradients_cond_while_grad_placeholder_5?
;gradients_cond_grad_gradients_cond_while_grad_placeholder_6?
;gradients_cond_grad_gradients_cond_while_grad_placeholder_7?
;gradients_cond_grad_gradients_cond_while_grad_placeholder_8?
;gradients_cond_grad_gradients_cond_while_grad_placeholder_9@
<gradients_cond_grad_gradients_cond_while_grad_placeholder_10
gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2write_tensorlistsetitem_grad_zeros_like_cond_while_selectv2q
mgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_selectv2_cond_while_tile_1б
Ьgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_broadcastgradientargs_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shapeг
Юgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_broadcastgradientargs_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shape_1е
аgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_broadcastgradientargs_1_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shape_2е
аgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_broadcastgradientargs_1_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shape_3q
mgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_selectv2_cond_while_tile_2б
Ьgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_broadcastgradientargs_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shapeг
Юgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_broadcastgradientargs_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shape_1е
аgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_broadcastgradientargs_1_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shape_2е
аgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_broadcastgradientargs_1_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shape_3m
igradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_selectv2_cond_while_tileЭ
Шgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_broadcastgradientargs_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_shapeб
Ьgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_broadcastgradientargs_1_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_shape_2g
cgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_2_grad_mul_cond_while_tanh_1l
hgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_2_grad_mul_1_cond_while_sigmoid_2Ч
Тgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_1_grad_broadcastgradientargs_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_1_grad_shapeЩ
Фgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_1_grad_broadcastgradientargs_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_1_grad_shape_1l
hgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_grad_mul_cond_while_placeholder_4j
fgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_grad_mul_1_cond_while_sigmoid_1e
agradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_1_grad_mul_cond_while_tanhj
fgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_1_grad_mul_1_cond_while_sigmoidУ
Оgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_grad_broadcastgradientargs_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_grad_shapeХ
Рgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_grad_broadcastgradientargs_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_grad_shape_1t
pgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_matmul_grad_matmul_cond_while_matmul_kernel_0
gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_matmul_grad_matmul_1_cond_while_tensorarrayv2read_tensorlistgetitem
~gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_matmul_1_grad_matmul_cond_while_matmul_1_recurrent_kernel_0v
rgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_matmul_1_grad_matmul_1_cond_while_placeholder_3
gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistreserve_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistelementshape
gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistreserve_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistlength
gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistsetitem_cond_while_placeholder:
6gradients_cond_grad_gradients_cond_while_grad_identity<
8gradients_cond_grad_gradients_cond_while_grad_identity_1<
8gradients_cond_grad_gradients_cond_while_grad_identity_2<
8gradients_cond_grad_gradients_cond_while_grad_identity_3<
8gradients_cond_grad_gradients_cond_while_grad_identity_4<
8gradients_cond_grad_gradients_cond_while_grad_identity_5<
8gradients_cond_grad_gradients_cond_while_grad_identity_6<
8gradients_cond_grad_gradients_cond_while_grad_identity_7<
8gradients_cond_grad_gradients_cond_while_grad_identity_8<
8gradients_cond_grad_gradients_cond_while_grad_identity_9=
9gradients_cond_grad_gradients_cond_while_grad_identity_10
gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2write_tensorlistsetitem_grad_zeros_like_tensorlistpopbackq
mgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_selectv2_tensorlistpopback~
zgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_broadcastgradientargs_tensorlistpopback
|gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_broadcastgradientargs_tensorlistpopback_1
|gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_broadcastgradientargs_1_tensorlistpopback
~gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_broadcastgradientargs_1_tensorlistpopback_1q
mgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_selectv2_tensorlistpopback~
zgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_broadcastgradientargs_tensorlistpopback
|gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_broadcastgradientargs_tensorlistpopback_1
|gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_broadcastgradientargs_1_tensorlistpopback
~gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_broadcastgradientargs_1_tensorlistpopback_1o
kgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_selectv2_tensorlistpopback|
xgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_broadcastgradientargs_tensorlistpopback~
zgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_broadcastgradientargs_1_tensorlistpopbackg
cgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_2_grad_mul_tensorlistpopbacki
egradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_2_grad_mul_1_tensorlistpopbacky
ugradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_1_grad_broadcastgradientargs_tensorlistpopback{
wgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_1_grad_broadcastgradientargs_tensorlistpopback_1e
agradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_grad_mul_tensorlistpopbackg
cgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_grad_mul_1_tensorlistpopbackg
cgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_1_grad_mul_tensorlistpopbacki
egradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_1_grad_mul_1_tensorlistpopbackw
sgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_grad_broadcastgradientargs_tensorlistpopbacky
ugradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_grad_broadcastgradientargs_tensorlistpopback_1r
ngradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_matmul_grad_matmul_cond_while_matmul_kernelm
igradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_matmul_grad_matmul_1_tensorlistpopback
|gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_matmul_1_grad_matmul_cond_while_matmul_1_recurrent_kernelo
kgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_matmul_1_grad_matmul_1_tensorlistpopback
gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistreserve_tensorlistpopback
gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistreserve_tensorlistpopback_1
gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistsetitem_tensorlistpopbackЋ
Agradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_0Identity;gradients_cond_grad_gradients_cond_while_grad_placeholder_3*
T0*
_output_shapes
: М
Agradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_1Identity;gradients_cond_grad_gradients_cond_while_grad_placeholder_4*
T0*'
_output_shapes
:џџџџџџџџџМ
Agradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_2Identity;gradients_cond_grad_gradients_cond_while_grad_placeholder_5*
T0*'
_output_shapes
:џџџџџџџџџМ
Agradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_3Identity;gradients_cond_grad_gradients_cond_while_grad_placeholder_6*
T0*'
_output_shapes
:џџџџџџџџџЋ
Agradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_4Identity;gradients_cond_grad_gradients_cond_while_grad_placeholder_7*
T0*
_output_shapes
: Д
Agradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_5Identity;gradients_cond_grad_gradients_cond_while_grad_placeholder_8*
T0*
_output_shapes
:	@Г
Agradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_6Identity;gradients_cond_grad_gradients_cond_while_grad_placeholder_9*
T0*
_output_shapes

:@А
Agradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_7Identity<gradients_cond_grad_gradients_cond_while_grad_placeholder_10*
T0*
_output_shapes
:@у
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/zeros_like/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/zeros_like/TensorListPopBackTensorListPopBackgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2write_tensorlistsetitem_grad_zeros_like_cond_while_selectv2 gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/zeros_like/TensorListPopBack/element_shape:output:0*)
_output_shapes
: :џџџџџџџџџ*
element_dtype0Ы
wgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/zeros_like	ZerosLikegradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/zeros_like/TensorListPopBack:tensor:0*
T0*'
_output_shapes
:џџџџџџџџџЧ
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/TensorListSetItem/ConstConst*
_output_shapes
: *
dtype0*
value	B : Ќ
~gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/TensorListSetItemTensorListSetItemJgradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_0:output:0gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/TensorListSetItem/Const:output:0{gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/zeros_like:y:0*
_output_shapes
: *
element_dtype0:щшвУ
rgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/ShapeShapegradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/zeros_like/TensorListPopBack:tensor:0*
T0*
_output_shapes
::эЯЊ
~gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/TensorListGetItemTensorListGetItemJgradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_0:output:0gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/TensorListSetItem/Const:output:0{gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/Shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0
Xgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
{gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/SelectV2/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџИ
mgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/SelectV2/TensorListPopBackTensorListPopBackmgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_selectv2_cond_while_tile_1gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/SelectV2/TensorListPopBack/element_shape:output:0*)
_output_shapes
: :џџџџџџџџџ*
element_dtype0
Р
[gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/SelectV2SelectV2vgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/SelectV2/TensorListPopBack:tensor:0Jgradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_2:output:0agradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџТ
]gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/SelectV2_1SelectV2vgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/SelectV2/TensorListPopBack:tensor:0agradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/zeros:output:0Jgradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџд
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЅ
zgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs/TensorListPopBackTensorListPopBackЬgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_broadcastgradientargs_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shapegradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs/TensorListPopBack/element_shape:output:0*
_output_shapes

: :*
element_dtype0ж
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs/TensorListPopBack_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЋ
|gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs/TensorListPopBack_1TensorListPopBackЮgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_broadcastgradientargs_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shape_1gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs/TensorListPopBack_1/element_shape:output:0*
_output_shapes

: :*
element_dtype0У
hgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs/TensorListPopBack:tensor:0gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs/TensorListPopBack_1:tensor:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџў
Vgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/SumSumdgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/SelectV2:output:0mgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ў
Zgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/ReshapeReshape_gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Sum:output:0gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs/TensorListPopBack:tensor:0*
T0*'
_output_shapes
:џџџџџџџџџж
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs_1/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ­
|gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs_1/TensorListPopBackTensorListPopBackаgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_broadcastgradientargs_1_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shape_2gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs_1/TensorListPopBack/element_shape:output:0*
_output_shapes

: :*
element_dtype0и
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs_1/TensorListPopBack_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџБ
~gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs_1/TensorListPopBack_1TensorListPopBackаgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_broadcastgradientargs_1_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_shape_3gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs_1/TensorListPopBack_1/element_shape:output:0*
_output_shapes

: :*
element_dtype0Щ
jgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs_1BroadcastGradientArgsgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs_1/TensorListPopBack:tensor:0gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs_1/TensorListPopBack_1:tensor:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Xgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Sum_1Sumfgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/SelectV2_1:output:0ogradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs_1:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
\gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Reshape_1Reshapeagradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Sum_1:output:0gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs_1/TensorListPopBack:tensor:0*
T0*'
_output_shapes
:џџџџџџџџџ
Xgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
{gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/SelectV2/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџИ
mgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/SelectV2/TensorListPopBackTensorListPopBackmgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_selectv2_cond_while_tile_2gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/SelectV2/TensorListPopBack/element_shape:output:0*)
_output_shapes
: :џџџџџџџџџ*
element_dtype0
Р
[gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/SelectV2SelectV2vgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/SelectV2/TensorListPopBack:tensor:0Jgradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_3:output:0agradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџТ
]gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/SelectV2_1SelectV2vgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/SelectV2/TensorListPopBack:tensor:0agradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/zeros:output:0Jgradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџд
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЅ
zgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs/TensorListPopBackTensorListPopBackЬgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_broadcastgradientargs_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shapegradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs/TensorListPopBack/element_shape:output:0*
_output_shapes

: :*
element_dtype0ж
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs/TensorListPopBack_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЋ
|gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs/TensorListPopBack_1TensorListPopBackЮgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_broadcastgradientargs_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shape_1gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs/TensorListPopBack_1/element_shape:output:0*
_output_shapes

: :*
element_dtype0У
hgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs/TensorListPopBack:tensor:0gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs/TensorListPopBack_1:tensor:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџў
Vgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/SumSumdgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/SelectV2:output:0mgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ў
Zgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/ReshapeReshape_gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Sum:output:0gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs/TensorListPopBack:tensor:0*
T0*'
_output_shapes
:џџџџџџџџџж
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs_1/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ­
|gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs_1/TensorListPopBackTensorListPopBackаgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_broadcastgradientargs_1_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shape_2gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs_1/TensorListPopBack/element_shape:output:0*
_output_shapes

: :*
element_dtype0и
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs_1/TensorListPopBack_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџБ
~gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs_1/TensorListPopBack_1TensorListPopBackаgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_broadcastgradientargs_1_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_shape_3gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs_1/TensorListPopBack_1/element_shape:output:0*
_output_shapes

: :*
element_dtype0Щ
jgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs_1BroadcastGradientArgsgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs_1/TensorListPopBack:tensor:0gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs_1/TensorListPopBack_1:tensor:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Xgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Sum_1Sumfgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/SelectV2_1:output:0ogradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs_1:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(
\gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Reshape_1Reshapeagradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Sum_1:output:0gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs_1/TensorListPopBack:tensor:0*
T0*'
_output_shapes
:џџџџџџџџџЉ
<gradients/cond_grad/gradients/cond/while_grad/gradients/AddNAddNJgradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_1:output:0gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/TensorListGetItem:item:0*
N*
T0*T
_classJ
HFloc:@gradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_1*'
_output_shapes
:џџџџџџџџџ
Vgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    Ф
ygradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/SelectV2/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџА
kgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/SelectV2/TensorListPopBackTensorListPopBackigradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_selectv2_cond_while_tilegradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/SelectV2/TensorListPopBack/element_shape:output:0*)
_output_shapes
: :џџџџџџџџџ*
element_dtype0
В
Ygradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/SelectV2SelectV2tgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/SelectV2/TensorListPopBack:tensor:0Bgradients/cond_grad/gradients/cond/while_grad/gradients/AddN:sum:0_gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/zeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџД
[gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/SelectV2_1SelectV2tgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/SelectV2/TensorListPopBack:tensor:0_gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/zeros:output:0Bgradients/cond_grad/gradients/cond/while_grad/gradients/AddN:sum:0*
T0*'
_output_shapes
:џџџџџџџџџЉ
Xgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_1Shapegradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/zeros_like/TensorListPopBack:tensor:0*
T0*
_output_shapes
::эЯв
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/BroadcastGradientArgs/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
xgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/BroadcastGradientArgs/TensorListPopBackTensorListPopBackШgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_broadcastgradientargs_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_shapegradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/BroadcastGradientArgs/TensorListPopBack/element_shape:output:0*
_output_shapes

: :*
element_dtype0
fgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/BroadcastGradientArgs/TensorListPopBack:tensor:0agradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџј
Tgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/SumSumbgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/SelectV2:output:0kgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ј
Xgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/ReshapeReshape]gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Sum:output:0gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/BroadcastGradientArgs/TensorListPopBack:tensor:0*
T0*'
_output_shapes
:џџџџџџџџџЉ
Xgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_3Shapegradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/zeros_like/TensorListPopBack:tensor:0*
T0*
_output_shapes
::эЯд
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/BroadcastGradientArgs_1/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЅ
zgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/BroadcastGradientArgs_1/TensorListPopBackTensorListPopBackЬgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_broadcastgradientargs_1_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_shape_2gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/BroadcastGradientArgs_1/TensorListPopBack/element_shape:output:0*
_output_shapes

: :*
element_dtype0
hgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/BroadcastGradientArgs_1BroadcastGradientArgsgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/BroadcastGradientArgs_1/TensorListPopBack:tensor:0agradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_3:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџў
Vgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Sum_1Sumdgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/SelectV2_1:output:0mgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/BroadcastGradientArgs_1:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ў
Zgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Reshape_1Reshape_gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Sum_1:output:0gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/BroadcastGradientArgs_1/TensorListPopBack:tensor:0*
T0*'
_output_shapes
:џџџџџџџџџИ
>gradients/cond_grad/gradients/cond/while_grad/gradients/AddN_1AddNcgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Reshape:output:0agradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Reshape:output:0*
N*
T0*m
_classc
a_loc:@gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџМ
qgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Mul/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
cgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Mul/TensorListPopBackTensorListPopBackcgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_2_grad_mul_cond_while_tanh_1zgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Mul/TensorListPopBack/element_shape:output:0*)
_output_shapes
: :џџџџџџџџџ*
element_dtype0О
Qgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/MulMulDgradients/cond_grad/gradients/cond/while_grad/gradients/AddN_1:sum:0lgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Mul/TensorListPopBack:tensor:0*
T0*'
_output_shapes
:џџџџџџџџџО
sgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Mul_1/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЂ
egradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Mul_1/TensorListPopBackTensorListPopBackhgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_2_grad_mul_1_cond_while_sigmoid_2|gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Mul_1/TensorListPopBack/element_shape:output:0*)
_output_shapes
: :џџџџџџџџџ*
element_dtype0Т
Sgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Mul_1Mulngradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Mul_1/TensorListPopBack:tensor:0Dgradients/cond_grad/gradients/cond/while_grad/gradients/AddN_1:sum:0*
T0*'
_output_shapes
:џџџџџџџџџџ
Sgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/ShapeShapengradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Mul_1/TensorListPopBack:tensor:0*
T0*
_output_shapes
::эЯџ
Ugradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Shape_1Shapelgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Mul/TensorListPopBack:tensor:0*
T0*
_output_shapes
::эЯю
cgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs\gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Shape:output:0^gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџх
Qgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/SumSumUgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Mul:z:0hgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(є
Ugradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/ReshapeReshapeZgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Sum:output:0\gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Shape:output:0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџщ
Sgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Sum_1SumWgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Mul_1:z:0hgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(њ
Wgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Reshape_1Reshape\gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Sum_1:output:0^gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Shape_1:output:0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџю
]gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/Sigmoid_2_grad/SigmoidGradSigmoidGradngradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Mul_1/TensorListPopBack:tensor:0^gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџх
Wgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/Tanh_1_grad/TanhGradTanhGradlgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Mul/TensorListPopBack:tensor:0`gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџВ
>gradients/cond_grad/gradients/cond/while_grad/gradients/AddN_2AddNcgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Reshape:output:0[gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/Tanh_1_grad/TanhGrad:z:0*
N*
T0*m
_classc
a_loc:@gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџЯ
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/BroadcastGradientArgs/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
ugradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/BroadcastGradientArgs/TensorListPopBackTensorListPopBackТgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_1_grad_broadcastgradientargs_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_1_grad_shapegradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/BroadcastGradientArgs/TensorListPopBack/element_shape:output:0*
_output_shapes

: :*
element_dtype0б
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/BroadcastGradientArgs/TensorListPopBack_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
wgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/BroadcastGradientArgs/TensorListPopBack_1TensorListPopBackФgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_1_grad_broadcastgradientargs_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_1_grad_shape_1gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/BroadcastGradientArgs/TensorListPopBack_1/element_shape:output:0*
_output_shapes

: :*
element_dtype0Г
cgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs~gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/BroadcastGradientArgs/TensorListPopBack:tensor:0gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/BroadcastGradientArgs/TensorListPopBack_1:tensor:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџд
Qgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/SumSumDgradients/cond_grad/gradients/cond/while_grad/gradients/AddN_2:sum:0hgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ю
Ugradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/ReshapeReshapeZgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Sum:output:0~gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/BroadcastGradientArgs/TensorListPopBack:tensor:0*
T0*'
_output_shapes
:џџџџџџџџџж
Sgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Sum_1SumDgradients/cond_grad/gradients/cond/while_grad/gradients/AddN_2:sum:0hgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ѕ
Wgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Reshape_1Reshape\gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Sum_1:output:0gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/BroadcastGradientArgs/TensorListPopBack_1:tensor:0*
T0*'
_output_shapes
:џџџџџџџџџК
ogradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Mul/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
agradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Mul/TensorListPopBackTensorListPopBackhgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_grad_mul_cond_while_placeholder_4xgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Mul/TensorListPopBack/element_shape:output:0*)
_output_shapes
: :џџџџџџџџџ*
element_dtype0д
Ogradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/MulMul^gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Reshape:output:0jgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Mul/TensorListPopBack:tensor:0*
T0*'
_output_shapes
:џџџџџџџџџМ
qgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Mul_1/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
cgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Mul_1/TensorListPopBackTensorListPopBackfgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_grad_mul_1_cond_while_sigmoid_1zgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Mul_1/TensorListPopBack/element_shape:output:0*)
_output_shapes
: :џџџџџџџџџ*
element_dtype0и
Qgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Mul_1Mullgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Mul_1/TensorListPopBack:tensor:0^gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџћ
Qgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/ShapeShapelgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Mul_1/TensorListPopBack:tensor:0*
T0*
_output_shapes
::эЯћ
Sgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Shape_1Shapejgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Mul/TensorListPopBack:tensor:0*
T0*
_output_shapes
::эЯш
agradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/BroadcastGradientArgsBroadcastGradientArgsZgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Shape:output:0\gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџп
Ogradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/SumSumSgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Mul:z:0fgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ю
Sgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/ReshapeReshapeXgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Sum:output:0Zgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Shape:output:0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџу
Qgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Sum_1SumUgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Mul_1:z:0fgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(Ь
Ugradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Reshape_1ReshapeZgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Sum_1:output:0\gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Shape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџМ
qgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Mul/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
cgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Mul/TensorListPopBackTensorListPopBackagradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_1_grad_mul_cond_while_tanhzgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Mul/TensorListPopBack/element_shape:output:0*)
_output_shapes
: :џџџџџџџџџ*
element_dtype0к
Qgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/MulMul`gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Reshape_1:output:0lgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Mul/TensorListPopBack:tensor:0*
T0*'
_output_shapes
:џџџџџџџџџО
sgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Mul_1/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ 
egradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Mul_1/TensorListPopBackTensorListPopBackfgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_1_grad_mul_1_cond_while_sigmoid|gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Mul_1/TensorListPopBack/element_shape:output:0*)
_output_shapes
: :џџџџџџџџџ*
element_dtype0о
Sgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Mul_1Mulngradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Mul_1/TensorListPopBack:tensor:0`gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџџ
Sgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/ShapeShapengradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Mul_1/TensorListPopBack:tensor:0*
T0*
_output_shapes
::эЯџ
Ugradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Shape_1Shapelgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Mul/TensorListPopBack:tensor:0*
T0*
_output_shapes
::эЯю
cgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs\gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Shape:output:0^gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Shape_1:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџх
Qgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/SumSumUgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Mul:z:0hgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(є
Ugradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/ReshapeReshapeZgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Sum:output:0\gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Shape:output:0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџщ
Sgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Sum_1SumWgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Mul_1:z:0hgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(њ
Wgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Reshape_1Reshape\gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Sum_1:output:0^gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Shape_1:output:0*
T0*&
 _has_manual_control_dependencies(*'
_output_shapes
:џџџџџџџџџъ
]gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/Sigmoid_1_grad/SigmoidGradSigmoidGradlgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Mul_1/TensorListPopBack:tensor:0\gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџЙ
>gradients/cond_grad/gradients/cond/while_grad/gradients/AddN_3AddNegradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Reshape_1:output:0^gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Reshape_1:output:0*
N*
T0*o
_classe
caloc:@gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџь
[gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/Sigmoid_grad/SigmoidGradSigmoidGradngradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Mul_1/TensorListPopBack:tensor:0^gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџу
Ugradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/Tanh_grad/TanhGradTanhGradlgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Mul/TensorListPopBack:tensor:0`gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
Zgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/split_grad/concat/ConstConst*
_output_shapes
: *
dtype0*
value	B :
Tgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/split_grad/concatConcatV2_gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/Sigmoid_grad/SigmoidGrad:z:0agradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/Sigmoid_1_grad/SigmoidGrad:z:0Ygradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/Tanh_grad/TanhGrad:z:0agradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/Sigmoid_2_grad/SigmoidGrad:z:0cgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/split_grad/concat/Const:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ@ю
[gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/BiasAdd_grad/BiasAddGradBiasAddGrad]gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/split_grad/concat:output:0*
T0*
_output_shapes
:@Э
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/BroadcastGradientArgs/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
sgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/BroadcastGradientArgs/TensorListPopBackTensorListPopBackОgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_grad_broadcastgradientargs_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_grad_shapegradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/BroadcastGradientArgs/TensorListPopBack/element_shape:output:0*
_output_shapes

: :*
element_dtype0Я
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/BroadcastGradientArgs/TensorListPopBack_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
ugradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/BroadcastGradientArgs/TensorListPopBack_1TensorListPopBackРgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_grad_broadcastgradientargs_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_grad_shape_1gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/BroadcastGradientArgs/TensorListPopBack_1/element_shape:output:0*
_output_shapes

: :*
element_dtype0Ќ
agradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/BroadcastGradientArgsBroadcastGradientArgs|gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/BroadcastGradientArgs/TensorListPopBack:tensor:0~gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/BroadcastGradientArgs/TensorListPopBack_1:tensor:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџщ
Ogradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/SumSum]gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/split_grad/concat:output:0fgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/BroadcastGradientArgs:r0:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ш
Sgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/ReshapeReshapeXgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Sum:output:0|gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/BroadcastGradientArgs/TensorListPopBack:tensor:0*
T0*'
_output_shapes
:џџџџџџџџџ@ы
Qgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Sum_1Sum]gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/split_grad/concat:output:0fgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/BroadcastGradientArgs:r1:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
	keep_dims(ю
Ugradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Reshape_1ReshapeZgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Sum_1:output:0~gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/BroadcastGradientArgs/TensorListPopBack_1:tensor:0*
T0*'
_output_shapes
:џџџџџџџџџ@ќ
>gradients/cond_grad/gradients/cond/while_grad/gradients/AddN_4AddNJgradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_7:output:0dgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/BiasAdd_grad/BiasAddGrad:output:0*
N*
T0*T
_classJ
HFloc:@gradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_7*
_output_shapes
:@ѕ
Ugradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/MatMul_grad/MatMulMatMul\gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Reshape:output:0pgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_matmul_grad_matmul_cond_while_matmul_kernel_0*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_b(Т
wgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/MatMul_grad/MatMul_1/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЫ
igradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/MatMul_grad/MatMul_1/TensorListPopBackTensorListPopBackgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_matmul_grad_matmul_1_cond_while_tensorarrayv2read_tensorlistgetitemgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/MatMul_grad/MatMul_1/TensorListPopBack/element_shape:output:0**
_output_shapes
: :џџџџџџџџџ*
element_dtype0№
Wgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/MatMul_grad/MatMul_1MatMulrgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/MatMul_grad/MatMul_1/TensorListPopBack:tensor:0\gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Reshape:output:0*
T0*
_output_shapes
:	@*
transpose_a(
Wgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/MatMul_1_grad/MatMulMatMul^gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Reshape_1:output:0~gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_matmul_1_grad_matmul_cond_while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_b(Ф
ygradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/MatMul_1_grad/MatMul_1/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЙ
kgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/MatMul_1_grad/MatMul_1/TensorListPopBackTensorListPopBackrgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_matmul_1_grad_matmul_1_cond_while_placeholder_3gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/MatMul_1_grad/MatMul_1/TensorListPopBack/element_shape:output:0*)
_output_shapes
: :џџџџџџџџџ*
element_dtype0ѕ
Ygradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/MatMul_1_grad/MatMul_1MatMultgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/MatMul_1_grad/MatMul_1/TensorListPopBack:tensor:0^gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Reshape_1:output:0*
T0*
_output_shapes

:@*
transpose_a(щ
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListReserve/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListReserve/TensorListPopBackTensorListPopBackgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistreserve_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistelementshapeІgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListReserve/TensorListPopBack/element_shape:output:0*
_output_shapes

: :*
element_dtype0ы
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListReserve/TensorListPopBack_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListReserve/TensorListPopBack_1TensorListPopBackgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistreserve_cond_while_gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistlengthЈgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListReserve/TensorListPopBack_1/element_shape:output:0*
_output_shapes
: : *
element_dtype0
}gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListReserveTensorListReservegradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListReserve/TensorListPopBack:tensor:0gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListReserve/TensorListPopBack_1:tensor:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвщ
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListSetItem/TensorListPopBack/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListSetItem/TensorListPopBackTensorListPopBackgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistsetitem_cond_while_placeholderІgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListSetItem/TensorListPopBack/element_shape:output:0*
_output_shapes
: : *
element_dtype0з
}gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListSetItemTensorListSetItemgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListReserve:handle:0gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListSetItem/TensorListPopBack:tensor:0_gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/MatMul_grad/MatMul:product:0*
_output_shapes
: *
element_dtype0:щшвў
>gradients/cond_grad/gradients/cond/while_grad/gradients/AddN_5AddNJgradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_5:output:0agradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/MatMul_grad/MatMul_1:product:0*
N*
T0*T
_classJ
HFloc:@gradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_5*
_output_shapes
:	@М
>gradients/cond_grad/gradients/cond/while_grad/gradients/AddN_6AddNegradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Reshape_1:output:0agradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/MatMul_1_grad/MatMul:product:0*
N*
T0*o
_classe
caloc:@gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџџ
>gradients/cond_grad/gradients/cond/while_grad/gradients/AddN_7AddNJgradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_6:output:0cgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/MatMul_1_grad/MatMul_1:product:0*
N*
T0*T
_classJ
HFloc:@gradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_6*
_output_shapes

:@Ђ
>gradients/cond_grad/gradients/cond/while_grad/gradients/AddN_8AddNJgradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_4:output:0gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListSetItem:output_handle:0*
N*
T0*T
_classJ
HFloc:@gradients/cond_grad/gradients/cond/while_grad/gradients/grad_ys_4*
_output_shapes
: u
3gradients/cond_grad/gradients/cond/while_grad/add/yConst*
_output_shapes
: *
dtype0*
value	B :д
1gradients/cond_grad/gradients/cond/while_grad/addAddV29gradients_cond_grad_gradients_cond_while_grad_placeholder<gradients/cond_grad/gradients/cond/while_grad/add/y:output:0*
T0*
_output_shapes
: 
6gradients/cond_grad/gradients/cond/while_grad/IdentityIdentity5gradients/cond_grad/gradients/cond/while_grad/add:z:0*
T0*
_output_shapes
: Ђ
8gradients/cond_grad/gradients/cond/while_grad/Identity_1Identity;gradients_cond_grad_gradients_cond_while_grad_placeholder_1*
T0*
_output_shapes
: Ђ
8gradients/cond_grad/gradients/cond/while_grad/Identity_2Identity;gradients_cond_grad_gradients_cond_while_grad_placeholder_2*
T0*
_output_shapes
: і
8gradients/cond_grad/gradients/cond/while_grad/Identity_3Identitygradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: л
8gradients/cond_grad/gradients/cond/while_grad/Identity_4Identitycgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Reshape_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџМ
8gradients/cond_grad/gradients/cond/while_grad/Identity_5IdentityDgradients/cond_grad/gradients/cond/while_grad/gradients/AddN_6:sum:0*
T0*'
_output_shapes
:џџџџџџџџџМ
8gradients/cond_grad/gradients/cond/while_grad/Identity_6IdentityDgradients/cond_grad/gradients/cond/while_grad/gradients/AddN_3:sum:0*
T0*'
_output_shapes
:џџџџџџџџџЋ
8gradients/cond_grad/gradients/cond/while_grad/Identity_7IdentityDgradients/cond_grad/gradients/cond/while_grad/gradients/AddN_8:sum:0*
T0*
_output_shapes
: Д
8gradients/cond_grad/gradients/cond/while_grad/Identity_8IdentityDgradients/cond_grad/gradients/cond/while_grad/gradients/AddN_5:sum:0*
T0*
_output_shapes
:	@Г
8gradients/cond_grad/gradients/cond/while_grad/Identity_9IdentityDgradients/cond_grad/gradients/cond/while_grad/gradients/AddN_7:sum:0*
T0*
_output_shapes

:@А
9gradients/cond_grad/gradients/cond/while_grad/Identity_10IdentityDgradients/cond_grad/gradients/cond/while_grad/gradients/AddN_4:sum:0*
T0*
_output_shapes
:@"џ
ugradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_1_grad_broadcastgradientargs_tensorlistpopbackgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/BroadcastGradientArgs/TensorListPopBack:output_handle:0"
wgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_1_grad_broadcastgradientargs_tensorlistpopback_1gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/BroadcastGradientArgs/TensorListPopBack_1:output_handle:0"ћ
sgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_grad_broadcastgradientargs_tensorlistpopbackgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/BroadcastGradientArgs/TensorListPopBack:output_handle:0"џ
ugradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_add_grad_broadcastgradientargs_tensorlistpopback_1gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/BroadcastGradientArgs/TensorListPopBack_1:output_handle:0"ъ
kgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_matmul_1_grad_matmul_1_tensorlistpopback{gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/MatMul_1_grad/MatMul_1/TensorListPopBack:output_handle:0"ў
|gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_matmul_1_grad_matmul_cond_while_matmul_1_recurrent_kernel~gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_matmul_1_grad_matmul_cond_while_matmul_1_recurrent_kernel_0"ц
igradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_matmul_grad_matmul_1_tensorlistpopbackygradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/MatMul_grad/MatMul_1/TensorListPopBack:output_handle:0"т
ngradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_matmul_grad_matmul_cond_while_matmul_kernelpgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_matmul_grad_matmul_cond_while_matmul_kernel_0"о
egradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_1_grad_mul_1_tensorlistpopbackugradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Mul_1/TensorListPopBack:output_handle:0"к
cgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_1_grad_mul_tensorlistpopbacksgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_1_grad/Mul/TensorListPopBack:output_handle:0"о
egradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_2_grad_mul_1_tensorlistpopbackugradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Mul_1/TensorListPopBack:output_handle:0"к
cgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_2_grad_mul_tensorlistpopbacksgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_2_grad/Mul/TensorListPopBack:output_handle:0"к
cgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_grad_mul_1_tensorlistpopbacksgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Mul_1/TensorListPopBack:output_handle:0"ж
agradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_mul_grad_mul_tensorlistpopbackqgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/mul_grad/Mul/TensorListPopBack:output_handle:0"
|gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_broadcastgradientargs_1_tensorlistpopbackgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs_1/TensorListPopBack:output_handle:0"
~gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_broadcastgradientargs_1_tensorlistpopback_1gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs_1/TensorListPopBack_1:output_handle:0"
zgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_broadcastgradientargs_tensorlistpopbackgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs/TensorListPopBack:output_handle:0"
|gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_broadcastgradientargs_tensorlistpopback_1gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/BroadcastGradientArgs/TensorListPopBack_1:output_handle:0"ю
mgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_1_grad_selectv2_tensorlistpopback}gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/SelectV2/TensorListPopBack:output_handle:0"
|gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_broadcastgradientargs_1_tensorlistpopbackgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs_1/TensorListPopBack:output_handle:0"
~gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_broadcastgradientargs_1_tensorlistpopback_1gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs_1/TensorListPopBack_1:output_handle:0"
zgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_broadcastgradientargs_tensorlistpopbackgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs/TensorListPopBack:output_handle:0"
|gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_broadcastgradientargs_tensorlistpopback_1gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/BroadcastGradientArgs/TensorListPopBack_1:output_handle:0"ю
mgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_2_grad_selectv2_tensorlistpopback}gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/SelectV2/TensorListPopBack:output_handle:0"
zgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_broadcastgradientargs_1_tensorlistpopbackgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/BroadcastGradientArgs_1/TensorListPopBack:output_handle:0"
xgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_broadcastgradientargs_tensorlistpopbackgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/BroadcastGradientArgs/TensorListPopBack:output_handle:0"ъ
kgradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_selectv2_grad_selectv2_tensorlistpopback{gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/SelectV2/TensorListPopBack:output_handle:0"Д
gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistreserve_tensorlistpopbackgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListReserve/TensorListPopBack:output_handle:0"И
gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistreserve_tensorlistpopback_1Ёgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListReserve/TensorListPopBack_1:output_handle:0"Д
gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2read_tensorlistgetitem_grad_tensorlistsetitem_tensorlistpopbackgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListSetItem/TensorListPopBack:output_handle:0"Ј
gradients_cond_grad_gradients_cond_while_grad_gradients_cond_while_tensorarrayv2write_tensorlistsetitem_grad_zeros_like_tensorlistpopbackgradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Write/TensorListSetItem_grad/zeros_like/TensorListPopBack:output_handle:0"y
6gradients_cond_grad_gradients_cond_while_grad_identity?gradients/cond_grad/gradients/cond/while_grad/Identity:output:0"}
8gradients_cond_grad_gradients_cond_while_grad_identity_1Agradients/cond_grad/gradients/cond/while_grad/Identity_1:output:0"
9gradients_cond_grad_gradients_cond_while_grad_identity_10Bgradients/cond_grad/gradients/cond/while_grad/Identity_10:output:0"}
8gradients_cond_grad_gradients_cond_while_grad_identity_2Agradients/cond_grad/gradients/cond/while_grad/Identity_2:output:0"}
8gradients_cond_grad_gradients_cond_while_grad_identity_3Agradients/cond_grad/gradients/cond/while_grad/Identity_3:output:0"}
8gradients_cond_grad_gradients_cond_while_grad_identity_4Agradients/cond_grad/gradients/cond/while_grad/Identity_4:output:0"}
8gradients_cond_grad_gradients_cond_while_grad_identity_5Agradients/cond_grad/gradients/cond/while_grad/Identity_5:output:0"}
8gradients_cond_grad_gradients_cond_while_grad_identity_6Agradients/cond_grad/gradients/cond/while_grad/Identity_6:output:0"}
8gradients_cond_grad_gradients_cond_while_grad_identity_7Agradients/cond_grad/gradients/cond/while_grad/Identity_7:output:0"}
8gradients_cond_grad_gradients_cond_while_grad_identity_8Agradients/cond_grad/gradients/cond/while_grad/Identity_8:output:0"}
8gradients_cond_grad_gradients_cond_while_grad_identity_9Agradients/cond_grad/gradients/cond/while_grad/Identity_9:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
­: : : : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: :	@:@:@: : : : : : : : : : : : : : : : : : : : : : : : :	@: :@: : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :%!

_output_shapes
:	@:$	 

_output_shapes

:@: 


_output_shapes
:@:KG

_output_shapes
: 
-
_user_specified_namecond/while/SelectV2:IE

_output_shapes
: 
+
_user_specified_namecond/while/Tile_1:

_output_shapes
: 
}
_user_specified_nameeccond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape:

_output_shapes
: 

_user_specified_namegecond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_1:

_output_shapes
: 

_user_specified_namegecond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_2:

_output_shapes
: 

_user_specified_namegecond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_1_grad/Shape_3:IE

_output_shapes
: 
+
_user_specified_namecond/while/Tile_2:

_output_shapes
: 
}
_user_specified_nameeccond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape:

_output_shapes
: 

_user_specified_namegecond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_1:

_output_shapes
: 

_user_specified_namegecond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_2:

_output_shapes
: 

_user_specified_namegecond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_2_grad/Shape_3:GC

_output_shapes
: 
)
_user_specified_namecond/while/Tile:

_output_shapes
: 
{
_user_specified_namecacond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape:

_output_shapes
: 
}
_user_specified_nameeccond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/SelectV2_grad/Shape_2:IE

_output_shapes
: 
+
_user_specified_namecond/while/Tanh_1:LH

_output_shapes
: 
.
_user_specified_namecond/while/Sigmoid_2:

_output_shapes
: 
x
_user_specified_name`^cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Shape:

_output_shapes
: 
z
_user_specified_nameb`cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_1_grad/Shape_1:PL

_output_shapes
: 
2
_user_specified_namecond/while/Placeholder_4:LH

_output_shapes
: 
.
_user_specified_namecond/while/Sigmoid_1:GC

_output_shapes
: 
)
_user_specified_namecond/while/Tanh:J F

_output_shapes
: 
,
_user_specified_namecond/while/Sigmoid:!

_output_shapes
: 
v
_user_specified_name^\cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Shape:"

_output_shapes
: 
x
_user_specified_name`^cond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/add_grad/Shape_1:Y#U

_output_shapes
:	@
2
_user_specified_namecond/while/MatMul/kernel:f$b

_output_shapes
: 
H
_user_specified_name0.cond/while/TensorArrayV2Read/TensorListGetItem:d%`

_output_shapes

:@
>
_user_specified_name&$cond/while/MatMul_1/recurrent_kernel:P&L

_output_shapes
: 
2
_user_specified_namecond/while/Placeholder_3:Щ'Ф

_output_shapes
: 
Љ
_user_specified_namecond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListElementShape:У(О

_output_shapes
: 
Ѓ
_user_specified_namecond/while/gradients/cond_grad/gradients/cond/while_grad/gradients/cond/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListLength:N)J

_output_shapes
: 
0
_user_specified_namecond/while/Placeholder
рK
Ђ
(__forward_gpu_lstm_with_fallback_1316360

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_splitW

zeros_likeConst*
_output_shapes
:@*
dtype0*
valueB@*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Ю
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_0eacfb0a-82c4-46c0-8e97-a277e1646ed4*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1316185_1316361*
go_backwards( *

time_major( :T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
§
М
D__inference_lstm_10_layer_call_and_return_conditional_losses_1316363

inputs/
read_readvariableop_resource:	@0
read_1_readvariableop_resource:@,
read_2_readvariableop_resource:@

identity_3ЂRead/ReadVariableOpЂRead_1/ReadVariableOpЂRead_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	@*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	@t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:@*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:@*
dtype0Z

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@Ж
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference_standard_lstm_1316090i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџh
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
К
H
,__inference_masking_10_layer_call_fn_1316963

inputs
identityК
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_masking_10_layer_call_and_return_conditional_losses_1315933e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
,
а
while_body_1317088
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@v
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@o
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:џџџџџџџџџ@W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ш
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
:	@
 
_user_specified_namekernel:P	L

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@
<

_output_shapes
:@

_user_specified_namebias
ѕ

*__inference_dense_20_layer_call_fn_1318743

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1316381o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:'#
!
_user_specified_name	1318737:'#
!
_user_specified_name	1318739
М
Q
$__inference__update_step_xla_1285111
gradient
variable:	@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	@: *
	_noinline(:I E

_output_shapes
:	@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ы

і
E__inference_dense_21_layer_call_and_return_conditional_losses_1316397

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
§
М
D__inference_lstm_10_layer_call_and_return_conditional_losses_1318734

inputs/
read_readvariableop_resource:	@0
read_1_readvariableop_resource:@,
read_2_readvariableop_resource:@

identity_3ЂRead/ReadVariableOpЂRead_1/ReadVariableOpЂRead_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	@*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	@t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:@*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:@*
dtype0Z

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@Ж
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference_standard_lstm_1318461i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџh
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ё

Ц
while_cond_1317087
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1317087___redundant_placeholder05
1while_while_cond_1317087___redundant_placeholder15
1while_while_cond_1317087___redundant_placeholder25
1while_while_cond_1317087___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ы

і
E__inference_dense_21_layer_call_and_return_conditional_losses_1318774

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
,
а
while_body_1316004
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@v
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@o
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:џџџџџџџџџ@W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ш
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
:	@
 
_user_specified_namekernel:P	L

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@
<

_output_shapes
:@

_user_specified_namebias
ћK
Ђ
(__forward_gpu_lstm_with_fallback_1317444

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_splitW

zeros_likeConst*
_output_shapes
:@*
dtype0*
valueB@*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0з
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_73bf78a1-4efb-49eb-a5db-1e73e2d416e1*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1317269_1317445*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
ћK
Ђ
(__forward_gpu_lstm_with_fallback_1317873

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_splitW

zeros_likeConst*
_output_shapes
:@*
dtype0*
valueB@*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0з
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_47cdb55f-aa18-46f4-a131-7cc1fe3544f6*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1317698_1317874*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
­
L
$__inference__update_step_xla_1285131
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ёL
В
cond_false_1313669
cond_expanddims_mask

cond_matmul_1_init_h
cond_mul_init_c
cond_matmul_kernel"
cond_matmul_1_recurrent_kernel
cond_biasadd_bias
cond_transpose_inputs
cond_identity
cond_identity_1
cond_identity_2
cond_identity_3
cond_identity_4h
cond/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
cond/transpose	Transposecond_transpose_inputscond/transpose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџZ

cond/ShapeShapecond/transpose:y:0*
T0*
_output_shapes
::эЯb
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
cond/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
cond/ExpandDims
ExpandDimscond_expanddims_maskcond/ExpandDims/dim:output:0*
T0
*+
_output_shapes
:џџџџџџџџџj
cond/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
cond/transpose_1	Transposecond/ExpandDims:output:0cond/transpose_1/perm:output:0*
T0
*+
_output_shapes
:џџџџџџџџџk
 cond/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџС
cond/TensorArrayV2TensorListReserve)cond/TensorArrayV2/element_shape:output:0cond/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
:cond/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   я
,cond/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorcond/transpose:y:0Ccond/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвd
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: f
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
cond/strided_slice_1StridedSlicecond/transpose:y:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskz
cond/MatMulMatMulcond/strided_slice_1:output:0cond_matmul_kernel*
T0*'
_output_shapes
:џџџџџџџџџ@
cond/MatMul_1MatMulcond_matmul_1_init_hcond_matmul_1_recurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ@s
cond/addAddV2cond/MatMul:product:0cond/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@j
cond/BiasAddBiasAddcond/add:z:0cond_biasadd_bias*
T0*'
_output_shapes
:џџџџџџџџџ@V
cond/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х

cond/splitSplitcond/split/split_dim:output:0cond/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split^
cond/SigmoidSigmoidcond/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
cond/Sigmoid_1Sigmoidcond/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџf
cond/mulMulcond/Sigmoid_1:y:0cond_mul_init_c*
T0*'
_output_shapes
:џџџџџџџџџX
	cond/TanhTanhcond/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџd

cond/mul_1Mulcond/Sigmoid:y:0cond/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџc

cond/add_1AddV2cond/mul:z:0cond/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ`
cond/Sigmoid_2Sigmoidcond/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџU
cond/Tanh_1Tanhcond/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџh

cond/mul_2Mulcond/Sigmoid_2:y:0cond/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџs
"cond/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   c
!cond/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :д
cond/TensorArrayV2_1TensorListReserve+cond/TensorArrayV2_1/element_shape:output:0*cond/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвK
	cond/timeConst*
_output_shapes
: *
dtype0*
value	B : m
"cond/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџХ
cond/TensorArrayV2_2TensorListReserve+cond/TensorArrayV2_2/element_shape:output:0cond/strided_slice:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ
<cond/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ѕ
.cond/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorcond/transpose_1:y:0Econd/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:щшШ^
cond/zeros_like	ZerosLikecond/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџh
cond/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџY
cond/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р

cond/whileStatelessWhile cond/while/loop_counter:output:0&cond/while/maximum_iterations:output:0cond/time:output:0cond/TensorArrayV2_1:handle:0cond/zeros_like:y:0cond_matmul_1_init_hcond_mul_init_ccond/strided_slice:output:0<cond/TensorArrayUnstack/TensorListFromTensor:output_handle:0>cond/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0cond_matmul_kernelcond_matmul_1_recurrent_kernelcond_biasadd_bias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*В
_output_shapes
: : : : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : :	@:@:@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : * 
_read_only_resource_inputs
 *#
bodyR
cond_while_body_1313816*#
condR
cond_while_cond_1313815*u
output_shapesd
b: : : : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : :	@:@:@*
parallel_iterations 
5cond/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   х
'cond/TensorArrayV2Stack/TensorListStackTensorListStackcond/while:output:3>cond/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsm
cond/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџf
cond/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: f
cond/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB: 
cond/strided_slice_2StridedSlice0cond/TensorArrayV2Stack/TensorListStack:tensor:0#cond/strided_slice_2/stack:output:0%cond/strided_slice_2/stack_1:output:0%cond/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskj
cond/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ѕ
cond/transpose_2	Transpose0cond/TensorArrayV2Stack/TensorListStack:tensor:0cond/transpose_2/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ`
cond/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?j
cond/IdentityIdentitycond/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџg
cond/Identity_1Identitycond/transpose_2:y:0*
T0*+
_output_shapes
:џџџџџџџџџb
cond/Identity_2Identitycond/while:output:5*
T0*'
_output_shapes
:џџџџџџџџџb
cond/Identity_3Identitycond/while:output:6*
T0*'
_output_shapes
:џџџџџџџџџS
cond/Identity_4Identitycond/runtime:output:0*
T0*
_output_shapes
: "'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0"+
cond_identity_2cond/Identity_2:output:0"+
cond_identity_3cond/Identity_3:output:0"+
cond_identity_4cond/Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@:џџџџџџџџџ:M I
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias:TP
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЊH
Ю
=__inference___backward_gpu_lstm_with_fallback_1313936_1315024
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4%
!gradients_cond_grad_if_logicaland

gradients_cond_grad_if_cond!
gradients_cond_grad_if_cond_1!
gradients_cond_grad_if_cond_2!
gradients_cond_grad_if_cond_3!
gradients_cond_grad_if_cond_4!
gradients_cond_grad_if_inputs!
gradients_cond_grad_if_cond_5!
gradients_cond_grad_if_cond_6!
gradients_cond_grad_if_cond_7!
gradients_cond_grad_if_cond_8!
gradients_cond_grad_if_cond_9!
gradients_cond_grad_if_init_h!
gradients_cond_grad_if_init_c"
gradients_cond_grad_if_cond_10"
gradients_cond_grad_if_cond_11"
gradients_cond_grad_if_cond_12"
gradients_cond_grad_if_cond_13"
gradients_cond_grad_if_cond_14"
gradients_cond_grad_if_cond_15"
gradients_cond_grad_if_cond_16"
gradients_cond_grad_if_cond_17"
gradients_cond_grad_if_cond_18"
gradients_cond_grad_if_cond_19"
gradients_cond_grad_if_cond_20"
gradients_cond_grad_if_cond_21"
gradients_cond_grad_if_cond_22"
gradients_cond_grad_if_cond_23"
gradients_cond_grad_if_cond_24"
gradients_cond_grad_if_cond_25"
gradients_cond_grad_if_cond_26"
gradients_cond_grad_if_cond_27"
gradients_cond_grad_if_cond_28"
gradients_cond_grad_if_cond_29"
gradients_cond_grad_if_cond_30"
gradients_cond_grad_if_cond_31"
gradients_cond_grad_if_cond_32"
gradients_cond_grad_if_cond_33"
gradients_cond_grad_if_cond_34"
gradients_cond_grad_if_cond_35"
gradients_cond_grad_if_cond_36
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ђgradients/cond_grad/If^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: Ј
gradients/cond_grad/IfIf!gradients_cond_grad_if_logicalandgradients/grad_ys_0:output:0gradients/grad_ys_1:output:0gradients/grad_ys_2:output:0gradients/grad_ys_3:output:0gradients/grad_ys_4:output:0gradients_cond_grad_if_condgradients_cond_grad_if_cond_1gradients_cond_grad_if_cond_2gradients_cond_grad_if_cond_3gradients_cond_grad_if_cond_4gradients_cond_grad_if_inputsgradients_cond_grad_if_cond_5gradients_cond_grad_if_cond_6gradients_cond_grad_if_cond_7gradients_cond_grad_if_cond_8gradients_cond_grad_if_cond_9gradients_cond_grad_if_init_hgradients_cond_grad_if_init_cgradients_cond_grad_if_cond_10gradients_cond_grad_if_cond_11gradients_cond_grad_if_cond_12gradients_cond_grad_if_cond_13gradients_cond_grad_if_cond_14gradients_cond_grad_if_cond_15gradients_cond_grad_if_cond_16gradients_cond_grad_if_cond_17gradients_cond_grad_if_cond_18gradients_cond_grad_if_cond_19gradients_cond_grad_if_cond_20gradients_cond_grad_if_cond_21gradients_cond_grad_if_cond_22gradients_cond_grad_if_cond_23gradients_cond_grad_if_cond_24gradients_cond_grad_if_cond_25gradients_cond_grad_if_cond_26gradients_cond_grad_if_cond_27gradients_cond_grad_if_cond_28gradients_cond_grad_if_cond_29gradients_cond_grad_if_cond_30gradients_cond_grad_if_cond_31gradients_cond_grad_if_cond_32gradients_cond_grad_if_cond_33gradients_cond_grad_if_cond_34gradients_cond_grad_if_cond_35gradients_cond_grad_if_cond_36*
Tcond0
*8
Tin1
/2-*
Tout

2*
_lower_using_switch_merge(*m
_output_shapes[
Y:џџџџџџџџџ:џџџџџџџџџ:	@:@:@:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
else_branch#R!
cond_false_1313669_grad_1314138*l
output_shapes[
Y:џџџџџџџџџ:џџџџџџџџџ:	@:@:@:џџџџџџџџџ*1
then_branch"R 
cond_true_1313668_grad_1313948{
gradients/cond_grad/IdentityIdentitygradients/cond_grad/If:output:0*
T0*'
_output_shapes
:џџџџџџџџџ}
gradients/cond_grad/Identity_1Identitygradients/cond_grad/If:output:1*
T0*'
_output_shapes
:џџџџџџџџџu
gradients/cond_grad/Identity_2Identitygradients/cond_grad/If:output:2*
T0*
_output_shapes
:	@t
gradients/cond_grad/Identity_3Identitygradients/cond_grad/If:output:3*
T0*
_output_shapes

:@p
gradients/cond_grad/Identity_4Identitygradients/cond_grad/If:output:4*
T0*
_output_shapes
:@
gradients/cond_grad/Identity_5Identitygradients/cond_grad/If:output:5*
T0*,
_output_shapes
:џџџџџџџџџ{
IdentityIdentity'gradients/cond_grad/Identity_5:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџv

Identity_1Identity%gradients/cond_grad/Identity:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџx

Identity_2Identity'gradients/cond_grad/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџp

Identity_3Identity'gradients/cond_grad/Identity_2:output:0^NoOp*
T0*
_output_shapes
:	@o

Identity_4Identity'gradients/cond_grad/Identity_3:output:0^NoOp*
T0*
_output_shapes

:@k

Identity_5Identity'gradients/cond_grad/Identity_4:output:0^NoOp*
T0*
_output_shapes
:@;
NoOpNoOp^gradients/cond_grad/If*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*ё
_input_shapesп
м:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : :џџџџџџџџџ: : : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : *=
api_implements+)lstm_4b11e03b-047e-44a3-bf0d-d903c89a5c70*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1315023*
go_backwards( *

time_major( 20
gradients/cond_grad/Ifgradients/cond_grad/If:- )
'
_output_shapes
:џџџџџџџџџ:1-
+
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :B>

_output_shapes
: 
$
_user_specified_name
LogicalAnd:<8

_output_shapes
: 

_user_specified_namecond:<8

_output_shapes
: 

_user_specified_namecond:<8

_output_shapes
: 

_user_specified_namecond:<	8

_output_shapes
: 

_user_specified_namecond:<
8

_output_shapes
: 

_user_specified_namecond:TP
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namecond:<8

_output_shapes
: 

_user_specified_namecond:<8

_output_shapes
: 

_user_specified_namecond:<8

_output_shapes
: 

_user_specified_namecond:<8

_output_shapes
: 

_user_specified_namecond:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:<8

_output_shapes
: 

_user_specified_namecond:<8

_output_shapes
: 

_user_specified_namecond:<8

_output_shapes
: 

_user_specified_namecond:<8

_output_shapes
: 

_user_specified_namecond:<8

_output_shapes
: 

_user_specified_namecond:<8

_output_shapes
: 

_user_specified_namecond:<8

_output_shapes
: 

_user_specified_namecond:<8

_output_shapes
: 

_user_specified_namecond:<8

_output_shapes
: 

_user_specified_namecond:<8

_output_shapes
: 

_user_specified_namecond:<8

_output_shapes
: 

_user_specified_namecond:<8

_output_shapes
: 

_user_specified_namecond:<8

_output_shapes
: 

_user_specified_namecond:< 8

_output_shapes
: 

_user_specified_namecond:<!8

_output_shapes
: 

_user_specified_namecond:<"8

_output_shapes
: 

_user_specified_namecond:<#8

_output_shapes
: 

_user_specified_namecond:<$8

_output_shapes
: 

_user_specified_namecond:<%8

_output_shapes
: 

_user_specified_namecond:<&8

_output_shapes
: 

_user_specified_namecond:<'8

_output_shapes
: 

_user_specified_namecond:<(8

_output_shapes
: 

_user_specified_namecond:<)8

_output_shapes
: 

_user_specified_namecond:<*8

_output_shapes
: 

_user_specified_namecond:<+8

_output_shapes
: 

_user_specified_namecond:<,8

_output_shapes
: 

_user_specified_namecond:<-8

_output_shapes
: 

_user_specified_namecond
;
С
!__inference_standard_lstm_1316090

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџP
ShapeShapetranspose:y:0*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ@^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ@d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:џџџџџџџџџ@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@* 
_read_only_resource_inputs
 *
bodyR
while_body_1316004*
condR
while_cond_1316003*`
output_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_0eacfb0a-82c4-46c0-8e97-a277e1646ed4*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias

М
D__inference_lstm_10_layer_call_and_return_conditional_losses_1315898

inputs/
read_readvariableop_resource:	@0
read_1_readvariableop_resource:@,
read_2_readvariableop_resource:@

identity_3ЂRead/ReadVariableOpЂRead_1/ReadVariableOpЂRead_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	@*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	@t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:@*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:@*
dtype0Z

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@Ж
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference_standard_lstm_1315625i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџh
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџ: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Њ;
С
!__inference_standard_lstm_1317174

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџP
ShapeShapetranspose:y:0*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ@^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ@d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:џџџџџџџџџ@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@* 
_read_only_resource_inputs
 *
bodyR
while_body_1317088*
condR
while_cond_1317087*`
output_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_73bf78a1-4efb-49eb-a5db-1e73e2d416e1*
api_preferred_deviceCPU*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
Ё

Ц
while_cond_1317516
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1317516___redundant_placeholder05
1while_while_cond_1317516___redundant_placeholder15
1while_while_cond_1317516___redundant_placeholder25
1while_while_cond_1317516___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ё

Ц
while_cond_1315538
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1315538___redundant_placeholder05
1while_while_cond_1315538___redundant_placeholder15
1while_while_cond_1315538___redundant_placeholder25
1while_while_cond_1315538___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ё

Ц
while_cond_1315109
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1315109___redundant_placeholder05
1while_while_cond_1315109___redundant_placeholder15
1while_while_cond_1315109___redundant_placeholder25
1while_while_cond_1315109___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ъ
ч
=__inference___backward_gpu_lstm_with_fallback_1316658_1316834
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
::эЯЊ
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯЄ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯЈ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџХ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯ
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Є
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*b
_output_shapesP
N:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:б
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:џџџџџџџџџ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
::эЯХ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
::эЯЩ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:ј
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ђ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ѓ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ѓ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Ж
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:З
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:З
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Џ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	@Д
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:@\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: e
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@g
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:@Ъ
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::б
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:@е
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:@s
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:џџџџџџџџџt

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	@g

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:@c

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:@"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*ї
_input_shapesх
т:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::џџџџџџџџџ:џџџџџџџџџ: ::::::::: : : : *=
api_implements+)lstm_2d8db02e-7366-49a3-aeb4-eef0d5b6c5e4*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1316833*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ:1-
+
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_namestrided_slice:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:B	>

_output_shapes
:
"
_user_specified_name
CudnnRNN:W
S
,
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	transpose:WS
+
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
ExpandDims:YU
+
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameExpandDims_1:FB

_output_shapes

:
"
_user_specified_name
concat_1:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:EA

_output_shapes
: 
'
_user_specified_nameconcat_1/axis:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_8/perm:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:C?

_output_shapes
: 
%
_user_specified_nameconcat/axis
ы
Ж
)__inference_lstm_10_layer_call_fn_1316996
inputs_0
unknown:	@
	unknown_0:@
	unknown_1:@
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lstm_10_layer_call_and_return_conditional_losses_1315898o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_0:'#
!
_user_specified_name	1316988:'#
!
_user_specified_name	1316990:'#
!
_user_specified_name	1316992
Ь

і
E__inference_dense_20_layer_call_and_return_conditional_losses_1316381

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ы
Ж
)__inference_lstm_10_layer_call_fn_1316985
inputs_0
unknown:	@
	unknown_0:@
	unknown_1:@
identityЂStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lstm_10_layer_call_and_return_conditional_losses_1315469o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_0:'#
!
_user_specified_name	1316977:'#
!
_user_specified_name	1316979:'#
!
_user_specified_name	1316981
,
а
while_body_1316477
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@v
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@o
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:џџџџџџџџџ@W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ш
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
:	@
 
_user_specified_namekernel:P	L

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@
<

_output_shapes
:@

_user_specified_namebias
Ё

Ц
while_cond_1318374
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1318374___redundant_placeholder05
1while_while_cond_1318374___redundant_placeholder15
1while_while_cond_1318374___redundant_placeholder25
1while_while_cond_1318374___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
­
L
$__inference__update_step_xla_1285121
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
кЪ
ч
=__inference___backward_gpu_lstm_with_fallback_1317269_1317445
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
::эЯЊ
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯЄ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯЈ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџХ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯ
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:Ђ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:­
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*k
_output_shapesY
W:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:к
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
::эЯХ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
::эЯЩ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:ј
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ђ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ѓ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ѓ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Ж
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:З
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:З
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Џ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	@Д
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:@\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: e
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@g
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:@Ъ
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::б
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:@е
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:@|
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџt

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	@g

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:@c

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:@"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapesї
є:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ::џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::џџџџџџџџџ:џџџџџџџџџ: ::::::::: : : : *=
api_implements+)lstm_73bf78a1-4efb-49eb-a5db-1e73e2d416e1*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1317444*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ:1-
+
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_namestrided_slice:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:^Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
CudnnRNN:B	>

_output_shapes
:
"
_user_specified_name
CudnnRNN:`
\
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	transpose:WS
+
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
ExpandDims:YU
+
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameExpandDims_1:FB

_output_shapes

:
"
_user_specified_name
concat_1:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:EA

_output_shapes
: 
'
_user_specified_nameconcat_1/axis:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_8/perm:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:C?

_output_shapes
: 
%
_user_specified_nameconcat/axis
;
С
!__inference_standard_lstm_1318461

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџP
ShapeShapetranspose:y:0*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ@^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ@d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:џџџџџџџџџ@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@* 
_read_only_resource_inputs
 *
bodyR
while_body_1318375*
condR
while_cond_1318374*`
output_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_7f07c2bf-ca9b-4883-99d3-8a7b5b1d9c08*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
Й
P
$__inference__update_step_xla_1285126
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:: *
	_noinline(:H D

_output_shapes

:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Й
P
$__inference__update_step_xla_1285136
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:: *
	_noinline(:H D

_output_shapes

:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ё

Ц
while_cond_1316476
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1316476___redundant_placeholder05
1while_while_cond_1316476___redundant_placeholder15
1while_while_cond_1316476___redundant_placeholder25
1while_while_cond_1316476___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
рK
Ђ
(__forward_gpu_lstm_with_fallback_1318302

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_splitW

zeros_likeConst*
_output_shapes
:@*
dtype0*
valueB@*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0Ю
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_771db2a2-7bde-4d3a-9c4a-c316a68cc663*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1318127_1318303*
go_backwards( *

time_major( :T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
;
С
!__inference_standard_lstm_1318032

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџP
ShapeShapetranspose:y:0*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ@^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ@d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:џџџџџџџџџ@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@* 
_read_only_resource_inputs
 *
bodyR
while_body_1317946*
condR
while_cond_1317945*`
output_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_771db2a2-7bde-4d3a-9c4a-c316a68cc663*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
ѕ

cond_false_1313669_grad_1314138?
;gradients_cond_grad_gradients_grad_ys_0_gradients_grad_ys_0?
;gradients_cond_grad_gradients_grad_ys_1_gradients_grad_ys_1?
;gradients_cond_grad_gradients_grad_ys_2_gradients_grad_ys_2?
;gradients_cond_grad_gradients_grad_ys_3_gradients_grad_ys_3?
;gradients_cond_grad_gradients_grad_ys_4_gradients_grad_ys_4i
egradients_cond_grad_gradients_cond_strided_slice_2_grad_shape_cond_tensorarrayv2stack_tensorliststack7
3gradients_cond_grad_gradients_zeros_like_cond_while9
5gradients_cond_grad_gradients_zeros_like_1_cond_while9
5gradients_cond_grad_gradients_zeros_like_2_cond_while9
5gradients_cond_grad_gradients_zeros_like_3_cond_while#
gradients_cond_grad_placeholder9
5gradients_cond_grad_gradients_zeros_like_4_cond_whileL
Hgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_whileN
Jgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_1N
Jgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_2N
Jgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_3%
!gradients_cond_grad_placeholder_1%
!gradients_cond_grad_placeholder_2N
Jgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_4N
Jgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_5N
Jgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_6N
Jgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_7N
Jgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_8N
Jgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_9O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_10O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_11O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_12O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_13O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_14O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_15O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_16O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_17O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_18O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_19O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_20O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_21O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_22O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_23O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_24O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_25O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_26O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_27O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_28O
Kgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_29h
dgradients_cond_grad_gradients_cond_tensorarrayunstack_tensorlistfromtensor_grad_shape_cond_transpose 
gradients_cond_grad_identity"
gradients_cond_grad_identity_1"
gradients_cond_grad_identity_2"
gradients_cond_grad_identity_3"
gradients_cond_grad_identity_4"
gradients_cond_grad_identity_5Ђ
'gradients/cond_grad/gradients/grad_ys_0Identity;gradients_cond_grad_gradients_grad_ys_0_gradients_grad_ys_0*
T0*'
_output_shapes
:џџџџџџџџџІ
'gradients/cond_grad/gradients/grad_ys_1Identity;gradients_cond_grad_gradients_grad_ys_1_gradients_grad_ys_1*
T0*+
_output_shapes
:џџџџџџџџџЂ
'gradients/cond_grad/gradients/grad_ys_2Identity;gradients_cond_grad_gradients_grad_ys_2_gradients_grad_ys_2*
T0*'
_output_shapes
:џџџџџџџџџЂ
'gradients/cond_grad/gradients/grad_ys_3Identity;gradients_cond_grad_gradients_grad_ys_3_gradients_grad_ys_3*
T0*'
_output_shapes
:џџџџџџџџџ
'gradients/cond_grad/gradients/grad_ys_4Identity;gradients_cond_grad_gradients_grad_ys_4_gradients_grad_ys_4*
T0*
_output_shapes
: Й
Ngradients/cond_grad/gradients/cond/strided_slice_2_grad/Shape/OptionalGetValueOptionalGetValueegradients_cond_grad_gradients_cond_strided_slice_2_grad_shape_cond_tensorarrayv2stack_tensorliststack*+
_output_shapes
:џџџџџџџџџ**
output_shapes
:џџџџџџџџџ*
output_types
2ж
=gradients/cond_grad/gradients/cond/strided_slice_2_grad/ShapeShape[gradients/cond_grad/gradients/cond/strided_slice_2_grad/Shape/OptionalGetValue:components:0*
T0*
_output_shapes
::эЯЁ
Ngradients/cond_grad/gradients/cond/strided_slice_2_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
Lgradients/cond_grad/gradients/cond/strided_slice_2_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
Pgradients/cond_grad/gradients/cond/strided_slice_2_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:М
Hgradients/cond_grad/gradients/cond/strided_slice_2_grad/StridedSliceGradStridedSliceGradFgradients/cond_grad/gradients/cond/strided_slice_2_grad/Shape:output:0Wgradients/cond_grad/gradients/cond/strided_slice_2_grad/StridedSliceGrad/begin:output:0Ugradients/cond_grad/gradients/cond/strided_slice_2_grad/StridedSliceGrad/end:output:0Ygradients/cond_grad/gradients/cond/strided_slice_2_grad/StridedSliceGrad/strides:output:00gradients/cond_grad/gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask 
Kgradients/cond_grad/gradients/cond/transpose_2_grad/InvertPermutation/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          Ь
Egradients/cond_grad/gradients/cond/transpose_2_grad/InvertPermutationInvertPermutationTgradients/cond_grad/gradients/cond/transpose_2_grad/InvertPermutation/Const:output:0*
_output_shapes
:§
=gradients/cond_grad/gradients/cond/transpose_2_grad/transpose	Transpose0gradients/cond_grad/gradients/grad_ys_1:output:0Igradients/cond_grad/gradients/cond/transpose_2_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџм
"gradients/cond_grad/gradients/AddNAddNQgradients/cond_grad/gradients/cond/strided_slice_2_grad/StridedSliceGrad:output:0Agradients/cond_grad/gradients/cond/transpose_2_grad/transpose:y:0*
N*
T0*[
_classQ
OMloc:@gradients/cond_grad/gradients/cond/strided_slice_2_grad/StridedSliceGrad*+
_output_shapes
:џџџџџџџџџО
mgradients/cond_grad/gradients/cond/TensorArrayV2Stack/TensorListStack_grad/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ы
_gradients/cond_grad/gradients/cond/TensorArrayV2Stack/TensorListStack_grad/TensorListFromTensorTensorListFromTensor(gradients/cond_grad/gradients/AddN:sum:0vgradients/cond_grad/gradients/cond/TensorArrayV2Stack/TensorListStack_grad/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвъ
9gradients/cond_grad/gradients/zeros_like/OptionalGetValueOptionalGetValue3gradients_cond_grad_gradients_zeros_like_cond_while*'
_output_shapes
:џџџџџџџџџ*&
output_shapes
:џџџџџџџџџ*
output_types
2Џ
(gradients/cond_grad/gradients/zeros_like	ZerosLikeFgradients/cond_grad/gradients/zeros_like/OptionalGetValue:components:0*
T0*'
_output_shapes
:џџџџџџџџџЬ
;gradients/cond_grad/gradients/zeros_like_1/OptionalGetValueOptionalGetValue5gradients_cond_grad_gradients_zeros_like_1_cond_while*
_output_shapes
: *
output_shapes
: *
output_types
2Ђ
*gradients/cond_grad/gradients/zeros_like_1	ZerosLikeHgradients/cond_grad/gradients/zeros_like_1/OptionalGetValue:components:0*
T0*
_output_shapes
: о
;gradients/cond_grad/gradients/zeros_like_2/OptionalGetValueOptionalGetValue5gradients_cond_grad_gradients_zeros_like_2_cond_while*
_output_shapes
:	@*
output_shapes
:	@*
output_types
2Ћ
*gradients/cond_grad/gradients/zeros_like_2	ZerosLikeHgradients/cond_grad/gradients/zeros_like_2/OptionalGetValue:components:0*
T0*
_output_shapes
:	@м
;gradients/cond_grad/gradients/zeros_like_3/OptionalGetValueOptionalGetValue5gradients_cond_grad_gradients_zeros_like_3_cond_while*
_output_shapes

:@*
output_shapes

:@*
output_types
2Њ
*gradients/cond_grad/gradients/zeros_like_3	ZerosLikeHgradients/cond_grad/gradients/zeros_like_3/OptionalGetValue:components:0*
T0*
_output_shapes

:@д
;gradients/cond_grad/gradients/zeros_like_4/OptionalGetValueOptionalGetValue5gradients_cond_grad_gradients_zeros_like_4_cond_while*
_output_shapes
:@*
output_shapes
:@*
output_types
2І
*gradients/cond_grad/gradients/zeros_like_4	ZerosLikeHgradients/cond_grad/gradients/zeros_like_4/OptionalGetValue:components:0*
T0*
_output_shapes
:@|
:gradients/cond_grad/gradients/cond/while_grad/grad_counterConst*
_output_shapes
: *
dtype0*
value	B : 
Cgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/ConstConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџђ
Ngradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValueOptionalGetValueHgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while*
_output_shapes
: *
output_shapes
: *
output_types
2і
Pgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_1OptionalGetValueJgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_1*
_output_shapes
: *
output_shapes
: *
output_types
2і
Pgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_2OptionalGetValueJgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_2*
_output_shapes
: *
output_shapes
: *
output_types
2і
Pgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_3OptionalGetValueJgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_3*
_output_shapes
: *
output_shapes
: *
output_types
2і
Pgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_4OptionalGetValueJgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_4*
_output_shapes
: *
output_shapes
: *
output_types
2і
Pgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_5OptionalGetValueJgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_5*
_output_shapes
: *
output_shapes
: *
output_types
2і
Pgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_6OptionalGetValueJgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_6*
_output_shapes
: *
output_shapes
: *
output_types
2і
Pgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_7OptionalGetValueJgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_7*
_output_shapes
: *
output_shapes
: *
output_types
2і
Pgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_8OptionalGetValueJgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_8*
_output_shapes
: *
output_shapes
: *
output_types
2і
Pgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_9OptionalGetValueJgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_9*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_10OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_10*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_11OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_11*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_12OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_12*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_13OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_13*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_14OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_14*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_15OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_15*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_16OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_16*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_17OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_17*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_18OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_18*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_19OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_19*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_20OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_20*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_21OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_21*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_22OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_22*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_23OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_23*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_24OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_24*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_25OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_25*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_26OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_26*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_27OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_27*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_28OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_28*
_output_shapes
: *
output_shapes
: *
output_types
2ј
Qgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_29OptionalGetValueKgradients_cond_grad_gradients_cond_while_grad_cond_while_grad_cond_while_29*
_output_shapes
: *
output_shapes
: *
output_types
2"
=gradients/cond_grad/gradients/cond/while_grad/cond/while_gradStatelessWhileCgradients/cond_grad/gradients/cond/while_grad/grad_counter:output:0Lgradients/cond_grad/gradients/cond/while_grad/cond/while_grad/Const:output:0[gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue:components:0ogradients/cond_grad/gradients/cond/TensorArrayV2Stack/TensorListStack_grad/TensorListFromTensor:output_handle:0,gradients/cond_grad/gradients/zeros_like:y:00gradients/cond_grad/gradients/grad_ys_2:output:00gradients/cond_grad/gradients/grad_ys_3:output:0.gradients/cond_grad/gradients/zeros_like_1:y:0.gradients/cond_grad/gradients/zeros_like_2:y:0.gradients/cond_grad/gradients/zeros_like_3:y:0.gradients/cond_grad/gradients/zeros_like_4:y:0]gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_1:components:0]gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_2:components:0]gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_3:components:0]gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_4:components:0]gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_5:components:0]gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_6:components:0]gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_7:components:0]gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_8:components:0]gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_9:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_10:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_11:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_12:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_13:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_14:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_15:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_16:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_17:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_18:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_19:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_20:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_21:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_22:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_23:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_24:components:0Hgradients/cond_grad/gradients/zeros_like_2/OptionalGetValue:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_25:components:0Hgradients/cond_grad/gradients/zeros_like_3/OptionalGetValue:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_26:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_27:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_28:components:0^gradients/cond_grad/gradients/cond/while_grad/cond/while_grad/OptionalGetValue_29:components:0*3
T.
,2**
_lower_using_switch_merge(*
_num_original_outputs**У
_output_shapesА
­: : : : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: :	@:@:@: : : : : : : : : : : : : : : : : : : : : : : : :	@: :@: : : : * 
_read_only_resource_inputs
 *0
body(R&
$cond_while_body_1313816_grad_1314184*:
cond2R0
.cond_while_cond_1313815_rewritten_grad_1314632*Т
output_shapesА
­: : : : :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: :	@:@:@: : : : : : : : : : : : : : : : : : : : : : : : :	@: :@: : : : *
parallel_iterations Ћ
6gradients/cond_grad/gradients/cond/while_grad/IdentityIdentityFgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:0*
T0*
_output_shapes
: ­
8gradients/cond_grad/gradients/cond/while_grad/Identity_1IdentityFgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:1*
T0*
_output_shapes
: ­
8gradients/cond_grad/gradients/cond/while_grad/Identity_2IdentityFgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:2*
T0*
_output_shapes
: ­
8gradients/cond_grad/gradients/cond/while_grad/Identity_3IdentityFgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:3*
T0*
_output_shapes
: О
8gradients/cond_grad/gradients/cond/while_grad/Identity_4IdentityFgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:4*
T0*'
_output_shapes
:џџџџџџџџџО
8gradients/cond_grad/gradients/cond/while_grad/Identity_5IdentityFgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:5*
T0*'
_output_shapes
:џџџџџџџџџО
8gradients/cond_grad/gradients/cond/while_grad/Identity_6IdentityFgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:6*
T0*'
_output_shapes
:џџџџџџџџџ­
8gradients/cond_grad/gradients/cond/while_grad/Identity_7IdentityFgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:7*
T0*
_output_shapes
: Ж
8gradients/cond_grad/gradients/cond/while_grad/Identity_8IdentityFgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:8*
T0*
_output_shapes
:	@Е
8gradients/cond_grad/gradients/cond/while_grad/Identity_9IdentityFgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:9*
T0*
_output_shapes

:@Г
9gradients/cond_grad/gradients/cond/while_grad/Identity_10IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:10*
T0*
_output_shapes
:@Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_11IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:11*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_12IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:12*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_13IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:13*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_14IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:14*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_15IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:15*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_16IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:16*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_17IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:17*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_18IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:18*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_19IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:19*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_20IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:20*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_21IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:21*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_22IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:22*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_23IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:23*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_24IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:24*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_25IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:25*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_26IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:26*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_27IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:27*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_28IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:28*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_29IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:29*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_30IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:30*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_31IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:31*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_32IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:32*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_33IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:33*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_34IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:34*
T0*
_output_shapes
: И
9gradients/cond_grad/gradients/cond/while_grad/Identity_35IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:35*
T0*
_output_shapes
:	@Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_36IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:36*
T0*
_output_shapes
: З
9gradients/cond_grad/gradients/cond/while_grad/Identity_37IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:37*
T0*
_output_shapes

:@Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_38IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:38*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_39IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:39*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_40IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:40*
T0*
_output_shapes
: Џ
9gradients/cond_grad/gradients/cond/while_grad/Identity_41IdentityGgradients/cond_grad/gradients/cond/while_grad/cond/while_grad:output:41*
T0*
_output_shapes
: в
fgradients/cond_grad/gradients/cond/TensorArrayUnstack/TensorListFromTensor_grad/Shape/OptionalGetValueOptionalGetValuedgradients_cond_grad_gradients_cond_tensorarrayunstack_tensorlistfromtensor_grad_shape_cond_transpose*,
_output_shapes
:џџџџџџџџџ*+
output_shapes
:џџџџџџџџџ*
output_types
2
Ugradients/cond_grad/gradients/cond/TensorArrayUnstack/TensorListFromTensor_grad/ShapeShapesgradients/cond_grad/gradients/cond/TensorArrayUnstack/TensorListFromTensor_grad/Shape/OptionalGetValue:components:0*
T0*
_output_shapes
::эЯЅ
[gradients/cond_grad/gradients/cond/TensorArrayUnstack/TensorListFromTensor_grad/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB:­
Zgradients/cond_grad/gradients/cond/TensorArrayUnstack/TensorListFromTensor_grad/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџЛ
Ugradients/cond_grad/gradients/cond/TensorArrayUnstack/TensorListFromTensor_grad/SliceSlice^gradients/cond_grad/gradients/cond/TensorArrayUnstack/TensorListFromTensor_grad/Shape:output:0dgradients/cond_grad/gradients/cond/TensorArrayUnstack/TensorListFromTensor_grad/Slice/begin:output:0cgradients/cond_grad/gradients/cond/TensorArrayUnstack/TensorListFromTensor_grad/Slice/size:output:0*
Index0*
T0*
_output_shapes
:ь
_gradients/cond_grad/gradients/cond/TensorArrayUnstack/TensorListFromTensor_grad/TensorListStackTensorListStackAgradients/cond_grad/gradients/cond/while_grad/Identity_7:output:0^gradients/cond_grad/gradients/cond/TensorArrayUnstack/TensorListFromTensor_grad/Slice:output:0*,
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elements
Igradients/cond_grad/gradients/cond/transpose_grad/InvertPermutation/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          Ш
Cgradients/cond_grad/gradients/cond/transpose_grad/InvertPermutationInvertPermutationRgradients/cond_grad/gradients/cond/transpose_grad/InvertPermutation/Const:output:0*
_output_shapes
:В
;gradients/cond_grad/gradients/cond/transpose_grad/transpose	Transposehgradients/cond_grad/gradients/cond/TensorArrayUnstack/TensorListFromTensor_grad/TensorListStack:tensor:0Ggradients/cond_grad/gradients/cond/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:џџџџџџџџџ
gradients/cond_grad/IdentityIdentityAgradients/cond_grad/gradients/cond/while_grad/Identity_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
gradients/cond_grad/Identity_1IdentityAgradients/cond_grad/gradients/cond/while_grad/Identity_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
gradients/cond_grad/Identity_2IdentityAgradients/cond_grad/gradients/cond/while_grad/Identity_8:output:0*
T0*
_output_shapes
:	@
gradients/cond_grad/Identity_3IdentityAgradients/cond_grad/gradients/cond/while_grad/Identity_9:output:0*
T0*
_output_shapes

:@
gradients/cond_grad/Identity_4IdentityBgradients/cond_grad/gradients/cond/while_grad/Identity_10:output:0*
T0*
_output_shapes
:@Ђ
gradients/cond_grad/Identity_5Identity?gradients/cond_grad/gradients/cond/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:џџџџџџџџџ"E
gradients_cond_grad_identity%gradients/cond_grad/Identity:output:0"I
gradients_cond_grad_identity_1'gradients/cond_grad/Identity_1:output:0"I
gradients_cond_grad_identity_2'gradients/cond_grad/Identity_2:output:0"I
gradients_cond_grad_identity_3'gradients/cond_grad/Identity_3:output:0"I
gradients_cond_grad_identity_4'gradients/cond_grad/Identity_4:output:0"I
gradients_cond_grad_identity_5'gradients/cond_grad/Identity_5:output:0*(
_construction_contextkEagerRuntime*я
_input_shapesн
к:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : :џџџџџџџџџ: : : : : :џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : :\ X
'
_output_shapes
:џџџџџџџџџ
-
_user_specified_namegradients/grad_ys_0:`\
+
_output_shapes
:џџџџџџџџџ
-
_user_specified_namegradients/grad_ys_1:\X
'
_output_shapes
:џџџџџџџџџ
-
_user_specified_namegradients/grad_ys_2:\X
'
_output_shapes
:џџџџџџџџџ
-
_user_specified_namegradients/grad_ys_3:KG

_output_shapes
: 
-
_user_specified_namegradients/grad_ys_4:_[

_output_shapes
: 
A
_user_specified_name)'cond/TensorArrayV2Stack/TensorListStack:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B	>

_output_shapes
: 
$
_user_specified_name
cond/while:2
.
,
_output_shapes
:џџџџџџџџџ:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B>

_output_shapes
: 
$
_user_specified_name
cond/while:B >

_output_shapes
: 
$
_user_specified_name
cond/while:B!>

_output_shapes
: 
$
_user_specified_name
cond/while:B">

_output_shapes
: 
$
_user_specified_name
cond/while:B#>

_output_shapes
: 
$
_user_specified_name
cond/while:B$>

_output_shapes
: 
$
_user_specified_name
cond/while:B%>

_output_shapes
: 
$
_user_specified_name
cond/while:B&>

_output_shapes
: 
$
_user_specified_name
cond/while:B'>

_output_shapes
: 
$
_user_specified_name
cond/while:B(>

_output_shapes
: 
$
_user_specified_name
cond/while:B)>

_output_shapes
: 
$
_user_specified_name
cond/while:B*>

_output_shapes
: 
$
_user_specified_name
cond/while:B+>

_output_shapes
: 
$
_user_specified_name
cond/while:F,B

_output_shapes
: 
(
_user_specified_namecond/transpose
кЪ
ч
=__inference___backward_gpu_lstm_with_fallback_1315291_1315467
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_43
/gradients_expanddims_2_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџd
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:џџџџџџџџџO
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 
!gradients/ExpandDims_2_grad/ShapeShape/gradients_expanddims_2_grad_shape_strided_slice*
T0*
_output_shapes
::эЯЊ
#gradients/ExpandDims_2_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_2_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯЄ
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯЈ
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџХ
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_2_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::эЯ
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:Ђ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
shrink_axis_maskc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:­
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*k
_output_shapesY
W:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:к
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
::эЯХ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
::эЯЩ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:@j
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:ј
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::ь
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:@№
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:№
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:я
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:я
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:ђ
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ђ
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ј
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      Ї
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ѓ
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ѓ
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:І
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:Ж
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:И
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:И
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:И
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:З
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:З
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:ч
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Џ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	@Д
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

:@\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: e
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:@g
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:@Ъ
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::б
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:@е
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:@|
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџt

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџv

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџf

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	@g

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

:@c

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes
:@"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapesї
є:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: :џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ::џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::џџџџџџџџџ:џџџџџџџџџ: ::::::::: : : : *=
api_implements+)lstm_40d6fd70-48b3-4841-ad66-b73fbb417689*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1315466*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ:1-
+
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_namestrided_slice:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
CudnnRNN:^Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
CudnnRNN:B	>

_output_shapes
:
"
_user_specified_name
CudnnRNN:`
\
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	transpose:WS
+
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
ExpandDims:YU
+
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameExpandDims_1:FB

_output_shapes

:
"
_user_specified_name
concat_1:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:EA

_output_shapes
: 
'
_user_specified_nameconcat_1/axis:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_8/perm:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:C?

_output_shapes
: 
%
_user_specified_nameconcat/axis
Ц
c
G__inference_masking_10_layer_call_and_return_conditional_losses_1316974

inputs
identityO

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    h
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџv
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(_
CastCastAny:output:0*

DstT0*

SrcT0
*+
_output_shapes
:џџџџџџџџџS
mulMulinputsCast:y:0*
T0*,
_output_shapes
:џџџџџџџџџr
SqueezeSqueezeAny:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims

џџџџџџџџџT
IdentityIdentitymul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё

Ц
while_cond_1316003
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1316003___redundant_placeholder05
1while_while_cond_1316003___redundant_placeholder15
1while_while_cond_1316003___redundant_placeholder25
1while_while_cond_1316003___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :џџџџџџџџџ:џџџџџџџџџ: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
С7
ъ
(__forward_gpu_lstm_with_fallback_1315023
inputs_0
init_h_0
init_c_0

kernel
recurrent_kernel
bias
mask

identity

identity_1

identity_2

identity_3

identity_4

logicaland

cond

cond_0

cond_1

cond_2

cond_3

inputs

cond_4

cond_5

cond_6

cond_7

cond_8

init_h

init_c

cond_9
cond_10
cond_11
cond_12
cond_13
cond_14
cond_15
cond_16
cond_17
cond_18
cond_19
cond_20
cond_21
cond_22
cond_23
cond_24
cond_25
cond_26
cond_27
cond_28
cond_29
cond_30
cond_31
cond_32
cond_33
cond_34
cond_35ЂcondG
ShapeShapemask*
T0
*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskS
CastCastmask*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџW
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :b
SumSumCast:y:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:џџџџџџџџџT
SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : V
SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :
SequenceMask/RangeRangeSequenceMask/Const:output:0strided_slice:output:0SequenceMask/Const_1:output:0*
_output_shapes
:f
SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
SequenceMask/ExpandDims
ExpandDimsSum:output:0$SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџ|
SequenceMask/CastCast SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ
SequenceMask/LessLessSequenceMask/Range:output:0SequenceMask/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ]
EqualEqualmaskSequenceMask/Less:z:0*
T0
*'
_output_shapes
:џџџџџџџџџV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       =
AllAll	Equal:z:0Const:output:0*
_output_shapes
: G

LogicalNot
LogicalNotmask*'
_output_shapes
:џџџџџџџџџY
All_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :c
All_1AllLogicalNot:y:0 All_1/reduction_indices:output:0*#
_output_shapes
:џџџџџџџџџQ
Const_1Const*
_output_shapes
:*
dtype0*
valueB: D
AnyAnyAll_1:output:0Const_1:output:0*
_output_shapes
: @
LogicalNot_1
LogicalNotAny:output:0*
_output_shapes
: P

LogicalAnd
LogicalAndAll:output:0LogicalNot_1:y:0*
_output_shapes
: 
cond_36IfLogicalAnd:z:0maskinit_h_0init_c_0kernelrecurrent_kernelbiasinputs_0*
Tcond0
*
Tin
	2
*6
Tout.
,2**
_lower_using_switch_merge(* 
_read_only_resource_inputs
 */
else_branch R
cond_false_1313669_rewritten*Б
output_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : *.
then_branchR
cond_true_1313668_rewritten]
cond/IdentityIdentitycond_36:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
cond/Identity_1Identitycond_36:output:1*
T0*+
_output_shapes
:џџџџџџџџџ_
cond/Identity_2Identitycond_36:output:2*
T0*'
_output_shapes
:џџџџџџџџџ_
cond/Identity_3Identitycond_36:output:3*
T0*'
_output_shapes
:џџџџџџџџџN
cond/Identity_4Identitycond_36:output:4*
T0*
_output_shapes
: e
IdentityIdentitycond/Identity:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџm

Identity_1Identitycond/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџi

Identity_2Identitycond/Identity_2:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџi

Identity_3Identitycond/Identity_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџX

Identity_4Identitycond/Identity_4:output:0^NoOp*
T0*
_output_shapes
: ,
NoOpNoOp^cond_36*
_output_shapes
 "
condcond_36:output:5"
cond_0cond_36:output:6"
cond_1cond_36:output:7"
cond_10cond_36:output:16"
cond_11cond_36:output:17"
cond_12cond_36:output:18"
cond_13cond_36:output:19"
cond_14cond_36:output:20"
cond_15cond_36:output:21"
cond_16cond_36:output:22"
cond_17cond_36:output:23"
cond_18cond_36:output:24"
cond_19cond_36:output:25"
cond_2cond_36:output:8"
cond_20cond_36:output:26"
cond_21cond_36:output:27"
cond_22cond_36:output:28"
cond_23cond_36:output:29"
cond_24cond_36:output:30"
cond_25cond_36:output:31"
cond_26cond_36:output:32"
cond_27cond_36:output:33"
cond_28cond_36:output:34"
cond_29cond_36:output:35"
cond_3cond_36:output:9"
cond_30cond_36:output:36"
cond_31cond_36:output:37"
cond_32cond_36:output:38"
cond_33cond_36:output:39"
cond_34cond_36:output:40"
cond_35cond_36:output:41"
cond_4cond_36:output:10"
cond_5cond_36:output:11"
cond_6cond_36:output:12"
cond_7cond_36:output:13"
cond_8cond_36:output:14"
cond_9cond_36:output:15"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"
inputsinputs_0"

logicalandLogicalAnd:z:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@:џџџџџџџџџ*=
api_implements+)lstm_4b11e03b-047e-44a3-bf0d-d903c89a5c70*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1313936_1315024*
go_backwards( *

time_major( 2
condcond_36:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias:MI
'
_output_shapes
:џџџџџџџџџ

_user_specified_namemask
Њ@
Э
*__inference_gpu_lstm_with_fallback_1316657

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_splitW

zeros_likeConst*
_output_shapes
:@*
dtype0*
valueB@*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes

:Ъ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_2d8db02e-7366-49a3-aeb4-eef0d5b6c5e4*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
,
а
while_body_1315539
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_bias
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ@v
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@o
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*'
_output_shapes
:џџџџџџџџџ@W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ш
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџl
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:џџџџџџџџџZ

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџg
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџf
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџb
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџW
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџk
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/mul_2:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ_
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ"*
while_biasadd_biaswhile_biasadd_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:GC

_output_shapes
:	@
 
_user_specified_namekernel:P	L

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@
<

_output_shapes
:@

_user_specified_namebias

О
D__inference_lstm_10_layer_call_and_return_conditional_losses_1317447
inputs_0/
read_readvariableop_resource:	@0
read_1_readvariableop_resource:@,
read_2_readvariableop_resource:@

identity_3ЂRead/ReadVariableOpЂRead_1/ReadVariableOpЂRead_2/ReadVariableOpK
ShapeShapeinputs_0*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџq
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	@*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	@t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

:@*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:@*
dtype0Z

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@И
PartitionedCallPartitionedCallinputs_0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *f
_output_shapesT
R:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference_standard_lstm_1317174i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџh
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџ: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_0:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Њ;
С
!__inference_standard_lstm_1315625

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџP
ShapeShapetranspose:y:0*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ@^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ@d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
BiasAddBiasAddadd:z:0bias*
T0*'
_output_shapes
:џџџџџџџџџ@Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:џџџџџџџџџS
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:џџџџџџџџџN
TanhTanhsplit:output:2*
T0*'
_output_shapes
:џџџџџџџџџU
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:џџџџџџџџџK
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџY
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*a
_output_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@* 
_read_only_resource_inputs
 *
bodyR
while_body_1315539*
condR
while_cond_1315538*`
output_shapesO
M: : : : :џџџџџџџџџ:џџџџџџџџџ: : :	@:@:@*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџX

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџX

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_ded2654b-8cb5-4033-bb74-72cb92ca8019*
api_preferred_deviceCPU*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias
ћK
Ђ
(__forward_gpu_lstm_with_fallback_1315895

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
splitSplitsplit/split_dim:output:0kernel*
T0*@
_output_shapes.
,:	:	:	:	*
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(::::*
	num_splitW

zeros_likeConst*
_output_shapes
:@*
dtype0*
valueB@*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:S
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0::::::::*
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes
:	Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       m
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes
:	[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:@a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:[
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:[
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:[
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:[
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
:[
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
:\

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
:\

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
:\

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
:\

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
:\

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
:\

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0з
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_2
ExpandDimsstrided_slice_0:output:0ExpandDims_2/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityExpandDims_2:output:0*
T0*+
_output_shapes
:џџџџџџџџџZ

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ\

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџI

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"!

cudnnrnn_0CudnnRNN:output_c:0"

cudnnrnn_1CudnnRNN:output:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:	@:@:@*=
api_implements+)lstm_ded2654b-8cb5-4033-bb74-72cb92ca8019*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1315720_1315896*
go_backwards( *

time_major( :] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_h:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinit_c:GC

_output_shapes
:	@
 
_user_specified_namekernel:PL

_output_shapes

:@
*
_user_specified_namerecurrent_kernel:@<

_output_shapes
:@

_user_specified_namebias"эL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*В
serving_default
B
input_116
serving_default_input_11:0џџџџџџџџџ<
dense_210
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:тЖ
ђ
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
к
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
Л
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
Л
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias"
_tf_keras_layer
Q
.0
/1
02
$3
%4
,5
-6"
trackable_list_wrapper
Q
.0
/1
02
$3
%4
,5
-6"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ч
6trace_0
7trace_12
*__inference_model_10_layer_call_fn_1316874
*__inference_model_10_layer_call_fn_1316893Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z6trace_0z7trace_1
§
8trace_0
9trace_12Ц
E__inference_model_10_layer_call_and_return_conditional_losses_1316404
E__inference_model_10_layer_call_and_return_conditional_losses_1316855Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z8trace_0z9trace_1
ЮBЫ
"__inference__wrapped_model_1315040input_11"
В
FullArgSpec
args

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Є
:
_variables
;_iterations
<_current_learning_rate
=_index_dict
>
_momentums
?_velocities
@_update_step_xla"
experimentalOptimizer
,
Aserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ц
Gtrace_02Щ
,__inference_masking_10_layer_call_fn_1316963
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zGtrace_0

Htrace_02ф
G__inference_masking_10_layer_call_and_return_conditional_losses_1316974
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zHtrace_0
5
.0
/1
02"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
Й

Istates
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ф
Otrace_0
Ptrace_1
Qtrace_2
Rtrace_32љ
)__inference_lstm_10_layer_call_fn_1316985
)__inference_lstm_10_layer_call_fn_1316996
)__inference_lstm_10_layer_call_fn_1317007
)__inference_lstm_10_layer_call_fn_1317018Ъ
УВП
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЂ

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zOtrace_0zPtrace_1zQtrace_2zRtrace_3
а
Strace_0
Ttrace_1
Utrace_2
Vtrace_32х
D__inference_lstm_10_layer_call_and_return_conditional_losses_1317447
D__inference_lstm_10_layer_call_and_return_conditional_losses_1317876
D__inference_lstm_10_layer_call_and_return_conditional_losses_1318305
D__inference_lstm_10_layer_call_and_return_conditional_losses_1318734Ъ
УВП
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЂ

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zStrace_0zTtrace_1zUtrace_2zVtrace_3
"
_generic_user_object
ј
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]_random_generator
^
state_size

.kernel
/recurrent_kernel
0bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
ф
dtrace_02Ч
*__inference_dense_20_layer_call_fn_1318743
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zdtrace_0
џ
etrace_02т
E__inference_dense_20_layer_call_and_return_conditional_losses_1318754
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zetrace_0
!:2dense_20/kernel
:2dense_20/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
ф
ktrace_02Ч
*__inference_dense_21_layer_call_fn_1318763
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zktrace_0
џ
ltrace_02т
E__inference_dense_21_layer_call_and_return_conditional_losses_1318774
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zltrace_0
!:2dense_21/kernel
:2dense_21/bias
+:)	@2lstm_10/lstm_cell/kernel
4:2@2"lstm_10/lstm_cell/recurrent_kernel
$:"@2lstm_10/lstm_cell/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
<
m0
n1
o2
p3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ъBч
*__inference_model_10_layer_call_fn_1316874input_11"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъBч
*__inference_model_10_layer_call_fn_1316893input_11"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_model_10_layer_call_and_return_conditional_losses_1316404input_11"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_model_10_layer_call_and_return_conditional_losses_1316855input_11"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 

;0
q1
r2
s3
t4
u5
v6
w7
x8
y9
z10
{11
|12
}13
~14"
trackable_list_wrapper
:	 2	iteration
: 2current_learning_rate
 "
trackable_dict_wrapper
Q
q0
s1
u2
w3
y4
{5
}6"
trackable_list_wrapper
Q
r0
t1
v2
x3
z4
|5
~6"
trackable_list_wrapper

trace_0
trace_1
trace_2
trace_3
trace_4
trace_5
trace_62М
$__inference__update_step_xla_1285111
$__inference__update_step_xla_1285116
$__inference__update_step_xla_1285121
$__inference__update_step_xla_1285126
$__inference__update_step_xla_1285131
$__inference__update_step_xla_1285136
$__inference__update_step_xla_1285141Џ
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0ztrace_0ztrace_1ztrace_2ztrace_3ztrace_4ztrace_5ztrace_6
гBа
%__inference_signature_wrapper_1316958input_11"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs

jinput_11
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBг
,__inference_masking_10_layer_call_fn_1316963inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
G__inference_masking_10_layer_call_and_return_conditional_losses_1316974inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
њBї
)__inference_lstm_10_layer_call_fn_1316985inputs_0"Н
ЖВВ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
)__inference_lstm_10_layer_call_fn_1316996inputs_0"Н
ЖВВ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
)__inference_lstm_10_layer_call_fn_1317007inputs"Н
ЖВВ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
)__inference_lstm_10_layer_call_fn_1317018inputs"Н
ЖВВ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_lstm_10_layer_call_and_return_conditional_losses_1317447inputs_0"Н
ЖВВ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_lstm_10_layer_call_and_return_conditional_losses_1317876inputs_0"Н
ЖВВ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_lstm_10_layer_call_and_return_conditional_losses_1318305inputs"Н
ЖВВ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_lstm_10_layer_call_and_return_conditional_losses_1318734inputs"Н
ЖВВ
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
5
.0
/1
02"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
Й2ЖГ
ЌВЈ
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Й2ЖГ
ЌВЈ
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
*__inference_dense_20_layer_call_fn_1318743inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
E__inference_dense_20_layer_call_and_return_conditional_losses_1318754inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
*__inference_dense_21_layer_call_fn_1318763inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
E__inference_dense_21_layer_call_and_return_conditional_losses_1318774inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
v
	variables
	keras_api

thresholds
true_positives
false_positives"
_tf_keras_metric
v
	variables
	keras_api

thresholds
true_positives
false_negatives"
_tf_keras_metric
0:.	@2Adam/m/lstm_10/lstm_cell/kernel
0:.	@2Adam/v/lstm_10/lstm_cell/kernel
9:7@2)Adam/m/lstm_10/lstm_cell/recurrent_kernel
9:7@2)Adam/v/lstm_10/lstm_cell/recurrent_kernel
):'@2Adam/m/lstm_10/lstm_cell/bias
):'@2Adam/v/lstm_10/lstm_cell/bias
&:$2Adam/m/dense_20/kernel
&:$2Adam/v/dense_20/kernel
 :2Adam/m/dense_20/bias
 :2Adam/v/dense_20/bias
&:$2Adam/m/dense_21/kernel
&:$2Adam/v/dense_21/kernel
 :2Adam/m/dense_21/bias
 :2Adam/v/dense_21/bias
яBь
$__inference__update_step_xla_1285111gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
$__inference__update_step_xla_1285116gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
$__inference__update_step_xla_1285121gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
$__inference__update_step_xla_1285126gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
$__inference__update_step_xla_1285131gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
$__inference__update_step_xla_1285136gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
$__inference__update_step_xla_1285141gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
$__inference__update_step_xla_1285111pjЂg
`Ђ]

gradient	@
52	Ђ
њ	@

p
` VariableSpec 
` єШ­аЗ=
Њ "
 
$__inference__update_step_xla_1285116nhЂe
^Ђ[

gradient@
41	Ђ
њ@

p
` VariableSpec 
`рнШ­аЗ=
Њ "
 
$__inference__update_step_xla_1285121f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
`р§РЪЗ=
Њ "
 
$__inference__update_step_xla_1285126nhЂe
^Ђ[

gradient
41	Ђ
њ

p
` VariableSpec 
`РЪЗ=
Њ "
 
$__inference__update_step_xla_1285131f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рЪЗ=
Њ "
 
$__inference__update_step_xla_1285136nhЂe
^Ђ[

gradient
41	Ђ
њ

p
` VariableSpec 
`ЇБфЭЗ=
Њ "
 
$__inference__update_step_xla_1285141f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рБфЭЗ=
Њ "
 
"__inference__wrapped_model_1315040v./0$%,-6Ђ3
,Ђ)
'$
input_11џџџџџџџџџ
Њ "3Њ0
.
dense_21"
dense_21џџџџџџџџџЌ
E__inference_dense_20_layer_call_and_return_conditional_losses_1318754c$%/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
*__inference_dense_20_layer_call_fn_1318743X$%/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџЌ
E__inference_dense_21_layer_call_and_return_conditional_losses_1318774c,-/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
*__inference_dense_21_layer_call_fn_1318763X,-/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџЮ
D__inference_lstm_10_layer_call_and_return_conditional_losses_1317447./0PЂM
FЂC
52
0-
inputs_0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Ю
D__inference_lstm_10_layer_call_and_return_conditional_losses_1317876./0PЂM
FЂC
52
0-
inputs_0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Н
D__inference_lstm_10_layer_call_and_return_conditional_losses_1318305u./0@Ђ=
6Ђ3
%"
inputsџџџџџџџџџ

 
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Н
D__inference_lstm_10_layer_call_and_return_conditional_losses_1318734u./0@Ђ=
6Ђ3
%"
inputsџџџџџџџџџ

 
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Ї
)__inference_lstm_10_layer_call_fn_1316985z./0PЂM
FЂC
52
0-
inputs_0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "!
unknownџџџџџџџџџЇ
)__inference_lstm_10_layer_call_fn_1316996z./0PЂM
FЂC
52
0-
inputs_0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "!
unknownџџџџџџџџџ
)__inference_lstm_10_layer_call_fn_1317007j./0@Ђ=
6Ђ3
%"
inputsџџџџџџџџџ

 
p

 
Њ "!
unknownџџџџџџџџџ
)__inference_lstm_10_layer_call_fn_1317018j./0@Ђ=
6Ђ3
%"
inputsџџџџџџџџџ

 
p 

 
Њ "!
unknownџџџџџџџџџД
G__inference_masking_10_layer_call_and_return_conditional_losses_1316974i4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ "1Ђ.
'$
tensor_0џџџџџџџџџ
 
,__inference_masking_10_layer_call_fn_1316963^4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ "&#
unknownџџџџџџџџџР
E__inference_model_10_layer_call_and_return_conditional_losses_1316404w./0$%,->Ђ;
4Ђ1
'$
input_11џџџџџџџџџ
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Р
E__inference_model_10_layer_call_and_return_conditional_losses_1316855w./0$%,->Ђ;
4Ђ1
'$
input_11џџџџџџџџџ
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
*__inference_model_10_layer_call_fn_1316874l./0$%,->Ђ;
4Ђ1
'$
input_11џџџџџџџџџ
p

 
Њ "!
unknownџџџџџџџџџ
*__inference_model_10_layer_call_fn_1316893l./0$%,->Ђ;
4Ђ1
'$
input_11џџџџџџџџџ
p 

 
Њ "!
unknownџџџџџџџџџЌ
%__inference_signature_wrapper_1316958./0$%,-BЂ?
Ђ 
8Њ5
3
input_11'$
input_11џџџџџџџџџ"3Њ0
.
dense_21"
dense_21џџџџџџџџџ