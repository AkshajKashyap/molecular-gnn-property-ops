def test_cli_import_works() -> None:
    from molgnn_ops.cli import app

    assert app is not None
