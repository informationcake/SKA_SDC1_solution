import os

from ska.sdc1.models.image_2d import Image2d


class TestImage2d:
    def test_preprocess(self, images_dir, test_image_name, pb_image_name):
        split_n = 3
        seg_files_expected = [
            test_image_name[:-5] + "_seg_{}.fits".format(i) for i in range(split_n ** 2)
        ]
        train_file_expected = test_image_name[:-5] + "_train.fits"
        test_image_path = os.path.join(images_dir, test_image_name)
        pb_image_path = os.path.join(images_dir, pb_image_name)

        # Before running preprocess, the segment and train files shouldn't exist
        for fp in seg_files_expected + [train_file_expected]:
            assert os.path.isfile(os.path.join(images_dir, fp)) is False
        image_2d = Image2d(560, test_image_path, pb_image_path, prep=False)
        image_2d.preprocess(split_n=split_n, overwrite=True)

        # Check files have been created
        for fp in seg_files_expected + [train_file_expected]:
            assert os.path.isfile(os.path.join(images_dir, fp))

        # Delete them again
        image_2d._delete_train()
        image_2d._delete_segments()

        # Verify
        for fp in seg_files_expected + [train_file_expected]:
            assert os.path.isfile(os.path.join(images_dir, fp)) is False
