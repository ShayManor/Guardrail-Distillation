find /workspace/.hf_home/hub/datasets--ShayManor--leftImg8bit_trainvaltest -name "*.png" | head -5
find /workspace/.hf_home/hub/datasets--ShayManor--gtFine_trainvaltest -name "*labelTrainIds*" | head -5
mkdir -p /workspace/data/cityscapes
ln -s /workspace/.hf_home/hub/datasets--ShayManor--leftImg8bit_trainvaltest/snapshots/*/leftImg8bit \
      /workspace/data/cityscapes/leftImg8bit
ln -s /workspace/.hf_home/hub/datasets--ShayManor--gtFine_trainvaltest/snapshots/*/gtFine \
      /workspace/data/cityscapes/gtFine