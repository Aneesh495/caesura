"""
Tests for Caesura CLI interface.
"""

import os
import tempfile
import pytest
from click.testing import CliRunner
from caesura.cli.main import cli
from caesura.vocab.vocab import create_byte_vocab
from caesura.models.bpe import BpeModel
from caesura.pipeline.tokenizer import Tokenizer
from caesura.io.format import BinarySerializer


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Caesura" in result.output
    assert "encode" in result.output
    assert "decode" in result.output
    assert "train" in result.output
    assert "benchmark" in result.output
    assert "verify" in result.output


def test_cli_encode_default():
    runner = CliRunner()
    result = runner.invoke(cli, ["encode", "hello world"])
    assert result.exit_code == 0
    assert "Encoded Tokens" in result.output


def test_cli_decode_default():
    runner = CliRunner()
    result = runner.invoke(cli, ["decode", "104, 101, 108, 108, 111"])
    assert result.exit_code == 0
    assert "Decoded Text" in result.output
    assert "hello" in result.output


def test_cli_verify():
    runner = CliRunner()
    result = runner.invoke(cli, ["verify"])
    assert result.exit_code == 0
    assert "Component Health Check" in result.output
    assert "zero defects" in result.output


def test_cli_benchmark():
    runner = CliRunner()
    result = runner.invoke(cli, ["benchmark", "-n", "1000"])
    assert result.exit_code == 0
    assert "Throughput Benchmark Results" in result.output
    assert "Tokens / Second" in result.output


def test_cli_train_and_inspect():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "corpus.txt")
        model_path = os.path.join(tmpdir, "model.caesura")
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write("apple banana apple orange banana apple grape\n" * 20)

        # Train command
        train_res = runner.invoke(cli, ["train", corpus_path, "-o", model_path, "-v", "300", "-f", "2"])
        assert train_res.exit_code == 0
        assert os.path.exists(model_path)

        # Inspect command
        inspect_res = runner.invoke(cli, ["inspect", model_path])
        assert inspect_res.exit_code == 0
        assert "Model Metadata" in inspect_res.output
        assert "BpeModel" in inspect_res.output

        # Encode with trained model
        enc_res = runner.invoke(cli, ["encode", "apple banana", "-m", model_path])
        assert enc_res.exit_code == 0
        assert "Encoded Tokens" in enc_res.output
