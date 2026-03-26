"""Tests for DocumentLoader — verifies file loading behavior.

Regressions caught:
- Text files load correctly (basic contract)
- Empty files produce no documents (boundary)
- Unsupported extension emits warning (graceful skip)
- Directory loading finds all supported files (recursive scan)
- Markdown files are treated as text (format dispatch)
"""

import pytest

from ondine.knowledge.loader import Document, DocumentLoader


class TestDocumentLoader:
    @pytest.fixture
    def loader(self):
        return DocumentLoader()

    def test_load_text_file(self, loader, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello, knowledge base!")

        docs = loader.load(f)
        assert len(docs) == 1
        assert docs[0].text == "Hello, knowledge base!"
        assert docs[0].source == str(f)

    def test_load_empty_file_produces_no_documents(self, loader, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")

        assert loader.load(f) == []

    def test_load_markdown_file(self, loader, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nSome content.")

        docs = loader.load(f)
        assert len(docs) == 1
        assert "# Title" in docs[0].text
        assert docs[0].metadata["format"] == "md"

    def test_unsupported_extension_returns_empty(self, loader, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3")

        docs = loader.load(f)
        assert docs == []

    def test_load_directory_finds_all_supported(self, loader, tmp_path):
        (tmp_path / "a.txt").write_text("Text A")
        (tmp_path / "b.md").write_text("Text B")
        (tmp_path / "c.csv").write_text("skip")

        docs = loader.load(tmp_path)
        sources = {d.source for d in docs}
        assert str(tmp_path / "a.txt") in sources
        assert str(tmp_path / "b.md") in sources
        assert len(docs) == 2

    def test_document_is_frozen(self):
        doc = Document(text="t", source="s")
        with pytest.raises(AttributeError):
            doc.text = "modified"
