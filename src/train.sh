sour_lang='ja'
tar_lang='en'
save_dir='./save'
knowledge_dimension=75
epochs=2
GAN_epochs=2
margin=1.0
kb_batchsize=256
cross_batchsize=32
GAN_batchsize=32
lr=0.001
cross_lr=0.0001
GAN_lr=0.00001
hidden_size=500
input_std=0.05
hid_std=0.05
dropout_rate=0
train_times=1
GAN_balance_factor=6.0
D_iter=1
clipping_parameter=0.01
rbf=0.0
CUDA_VISIBLE_DEVICES=0 python AKE.py --sour_lang $sour_lang --tar_lang $tar_lang --save_dir $save_dir \
 --knowledge_dimension $knowledge_dimension --epochs $epochs --GAN_epochs $GAN_epochs --margin $margin \
 --kb_batchsize $kb_batchsize --cross_batchsize $cross_batchsize --GAN_batchsize $GAN_batchsize --lr $lr \
 --cross_lr $cross_lr --GAN_lr $GAN_lr --hidden_size $hidden_size --input_std $input_std --hid_std $hid_std  \
 --dropout_rate $dropout_rate --train_times $train_times --GAN_balance_factor $GAN_balance_factor \
 --D_iter $D_iter --clipping_parameter $clipping_parameter --rbf $rbf
