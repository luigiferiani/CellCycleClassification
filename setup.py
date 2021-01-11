import setuptools

setuptools.setup(
    name='cellcycleclassification',
    version='0.0.2',
    description='Tools to classify at what stage of the cell cycle cells are',
    url='https://github.com/luigiferiani/CellCycleClassification',
    author='Luigi Feriani',
    author_email='l.feriani@lms.mrc.ac.uk',
    license='MIT',
    packages=setuptools.find_packages(),
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "create_dataset="
            + "cellcycleclassification.processing.create_annotations_dataset:"
            + "main",
            "annotate_dataset="
            + "cellcycleclassification.manual_annotation_gui."
            + "CellCycleAnnotator:"
            + "main",
            "classify="
            + "cellcycleclassification.video_processing."
            + "process_nuclitracked_videos:"
            + "main"
        ]
    },
    )
