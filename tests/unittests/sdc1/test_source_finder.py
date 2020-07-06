import os
import time

import pandas as pd

from ska.sdc1.utils.source_finder import SourceFinder


class TestSourceFinder:
    def test_find_sources(self, images_dir, test_image_name):
        image_path = os.path.join(images_dir, test_image_name)
        source_finder = SourceFinder(image_path)

        assert not source_finder._run_complete
        assert not os.path.exists(source_finder.get_bdsf_log_path())
        assert not os.path.isdir(source_finder.get_bdsf_out_path())

        source_finder.run()

        assert source_finder._run_complete
        assert os.path.exists(source_finder.get_bdsf_log_path())
        assert os.path.exists(source_finder.get_srl_path())
        assert os.path.exists(source_finder.get_gaul_path())
        assert os.path.isdir(source_finder.get_bdsf_out_path())

        source_finder.reset()

        assert not source_finder._run_complete
        assert not os.path.exists(source_finder.get_bdsf_log_path())
        assert not os.path.isdir(source_finder.get_bdsf_out_path())

    def test_retrieve_source_list(self, images_dir, test_image_name):
        image_path = os.path.join(images_dir, test_image_name)
        source_finder = SourceFinder(image_path)

        source_finder.run()

        source_df = source_finder.get_source_df()
        assert isinstance(source_df, pd.DataFrame)

        source_finder.reset()

        assert not source_finder._run_complete
        assert not os.path.exists(source_finder.get_bdsf_log_path())
        assert not os.path.isdir(source_finder.get_bdsf_out_path())
