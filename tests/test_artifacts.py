from image_captioning.captioner import missing_artifacts, required_artifact_paths


def test_required_artifact_contract_lists_three_files():
    names = {path.name for path in required_artifact_paths()}

    assert names == {"model.keras", "feature_extractor.keras", "tokenizer.pkl"}


def test_committed_demo_artifacts_are_present():
    assert missing_artifacts() == []
