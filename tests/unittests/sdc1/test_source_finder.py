from ska.sdc1.utils.source_finder import SourceFinder


class TestSourceFinder:
    def test_find_sources(self, image_path, image_name):
        source_finder = SourceFinder(image_path, image_name)
        source_finder.run()
        assert source_finder._run_complete
