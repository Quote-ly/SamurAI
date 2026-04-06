"""Tests for tools.fedramp_oscal — OSCAL lifecycle tools for FedRAMP compliance."""

import base64
import json
import re
import uuid
from unittest.mock import MagicMock, patch

import pytest

from tools.fedramp_oscal import (
    _pending_file_uploads,
    _pending_fedramp_cards,
    _parse_markdown_table,
    _find_control_in_catalog,
    _extract_control_ids_from_profile,
    OSCAL_VERSION,
    FEDRAMP_SYSTEM_NAME,
    FEDRAMP_SYSTEM_ID,
)

AUTHORIZED_EMAIL = "devin@virtualdojo.com"
UNAUTHORIZED_EMAIL = "hacker@example.com"
CONV_ID = "test-conv-oscal-456"
MOCK_TOKEN = "ghp_test_token_123"


@pytest.fixture(autouse=True)
def clear_pending():
    """Clear pending state before and after each test."""
    _pending_file_uploads.clear()
    _pending_fedramp_cards.clear()
    yield
    _pending_file_uploads.clear()
    _pending_fedramp_cards.clear()


def _mock_github_token():
    return patch("tools.github._github_token", return_value=MOCK_TOKEN)


def _mock_fedramp_profile():
    """Return a minimal FedRAMP Moderate baseline profile."""
    return {
        "profile": {
            "imports": [
                {
                    "include-controls": [
                        {"with-ids": ["ac-2", "si-4", "au-6"]},
                    ]
                }
            ]
        }
    }


def _mock_nist_catalog():
    """Return a minimal NIST 800-53 catalog for testing."""
    return {
        "catalog": {
            "groups": [
                {
                    "id": "ac",
                    "title": "Access Control",
                    "controls": [
                        {
                            "id": "ac-2",
                            "title": "Account Management",
                            "parts": [
                                {
                                    "name": "statement",
                                    "prose": "The organization manages information system accounts.",
                                },
                                {
                                    "name": "guidance",
                                    "prose": "Account management includes...",
                                },
                            ],
                            "params": [
                                {
                                    "id": "ac-2_prm_1",
                                    "label": "organization-defined types",
                                    "guidelines": [{"prose": "Define account types."}],
                                }
                            ],
                            "controls": [
                                {
                                    "id": "ac-2.1",
                                    "title": "Automated System Account Management",
                                    "parts": [],
                                }
                            ],
                        },
                    ],
                },
                {
                    "id": "si",
                    "title": "System and Information Integrity",
                    "controls": [
                        {
                            "id": "si-4",
                            "title": "Information System Monitoring",
                            "parts": [
                                {"name": "statement", "prose": "The organization monitors the information system."},
                            ],
                            "params": [],
                            "controls": [],
                        },
                    ],
                },
            ]
        }
    }


def _mock_ssp_json():
    """Return a minimal SSP JSON structure."""
    return {
        "system-security-plan": {
            "uuid": str(uuid.uuid4()),
            "metadata": {
                "title": "Test SSP",
                "last-modified": "2026-04-01T00:00:00Z",
                "version": "1.0",
                "oscal-version": OSCAL_VERSION,
            },
            "system-characteristics": {
                "system-name": FEDRAMP_SYSTEM_NAME,
                "description": "Test system",
                "security-sensitivity-level": "moderate",
                "system-information": {"information-types": []},
                "security-impact-level": {
                    "security-objective-confidentiality": "moderate",
                    "security-objective-integrity": "moderate",
                    "security-objective-availability": "moderate",
                },
                "status": {"state": "operational"},
                "authorization-boundary": {"description": "Test boundary"},
            },
            "system-implementation": {
                "users": [],
                "components": [
                    {
                        "uuid": str(uuid.uuid4()),
                        "type": "this-system",
                        "title": "Test System",
                        "description": "The primary system.",
                        "status": {"state": "operational"},
                    }
                ],
            },
            "control-implementation": {
                "description": "Test implementations",
                "implemented-requirements": [
                    {
                        "uuid": str(uuid.uuid4()),
                        "control-id": "ac-2",
                        "statements": [
                            {
                                "statement-id": "ac-2_smt",
                                "uuid": str(uuid.uuid4()),
                                "description": "Old AC-2 implementation text.",
                            }
                        ],
                    },
                    {
                        "uuid": str(uuid.uuid4()),
                        "control-id": "si-4",
                        "statements": [
                            {
                                "statement-id": "si-4_smt",
                                "uuid": str(uuid.uuid4()),
                                "description": "Old SI-4 implementation text.",
                            }
                        ],
                    },
                ],
            },
        }
    }


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestParseMarkdownTable:
    """Tests for _parse_markdown_table helper."""

    def test_basic_table(self):
        md = (
            "| Control ID | Description | Risk Level |\n"
            "| --- | --- | --- |\n"
            "| AC-2 | Account mgmt finding | High |\n"
            "| SI-4 | Monitoring gap | Moderate |\n"
        )
        rows = _parse_markdown_table(md)
        assert len(rows) == 2
        assert rows[0]["Control ID"] == "AC-2"
        assert rows[1]["Risk Level"] == "Moderate"

    def test_empty_table(self):
        md = "No table here.\nJust plain text."
        rows = _parse_markdown_table(md)
        assert rows == []

    def test_header_only(self):
        md = "| Col A | Col B |\n| --- | --- |"
        rows = _parse_markdown_table(md)
        assert rows == []


class TestFindControlInCatalog:
    """Tests for _find_control_in_catalog helper."""

    def test_find_existing_control(self):
        catalog = _mock_nist_catalog()
        ctrl = _find_control_in_catalog(catalog, "ac-2")
        assert ctrl is not None
        assert ctrl["title"] == "Account Management"

    def test_find_enhancement(self):
        catalog = _mock_nist_catalog()
        ctrl = _find_control_in_catalog(catalog, "ac-2.1")
        assert ctrl is not None
        assert ctrl["title"] == "Automated System Account Management"

    def test_control_not_found(self):
        catalog = _mock_nist_catalog()
        ctrl = _find_control_in_catalog(catalog, "zz-99")
        assert ctrl is None


class TestExtractControlIds:
    """Tests for _extract_control_ids_from_profile helper."""

    def test_extract_ids(self):
        profile = _mock_fedramp_profile()
        ids = _extract_control_ids_from_profile(profile)
        assert "ac-2" in ids
        assert "si-4" in ids
        assert "au-6" in ids

    def test_empty_profile(self):
        ids = _extract_control_ids_from_profile({"profile": {"imports": []}})
        assert ids == []


# ---------------------------------------------------------------------------
# oscal_generate_ssp
# ---------------------------------------------------------------------------


class TestGenerateSsp:
    """Tests for oscal_generate_ssp — OSCAL SSP generation."""

    def test_unauthorized(self):
        from tools.fedramp_oscal import oscal_generate_ssp

        result = oscal_generate_ssp.invoke({
            "conversation_id": CONV_ID,
            "user_email": UNAUTHORIZED_EMAIL,
        })
        assert "not authorized" in result

    @patch("tools.fedramp_oscal._commit_file", return_value="abc123sha")
    @patch("tools.fedramp_oscal._read_github_file", return_value=None)
    @patch("tools.fedramp_oscal._get_fedramp_profile")
    def test_generates_ssp_structure(self, mock_profile, mock_read, mock_commit):
        from tools.fedramp_oscal import oscal_generate_ssp

        mock_profile.return_value = _mock_fedramp_profile()

        result = oscal_generate_ssp.invoke({
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "OSCAL System Security Plan generated" in result
        assert "abc123sha" in result
        assert OSCAL_VERSION in result
        assert FEDRAMP_SYSTEM_NAME in result

        # Verify the committed JSON structure
        committed_json = json.loads(mock_commit.call_args[0][1])
        ssp = committed_json["system-security-plan"]
        assert "uuid" in ssp
        assert ssp["metadata"]["oscal-version"] == OSCAL_VERSION
        assert "system-characteristics" in ssp
        assert "system-implementation" in ssp
        assert "control-implementation" in ssp
        impl_reqs = ssp["control-implementation"]["implemented-requirements"]
        assert len(impl_reqs) == 3  # ac-2, si-4, au-6

    @patch("tools.fedramp_oscal._commit_file", return_value="sha456")
    @patch("tools.fedramp_oscal._get_fedramp_profile")
    @patch("tools.fedramp_oscal._read_github_file")
    def test_populates_from_markdown(self, mock_read, mock_profile, mock_commit):
        from tools.fedramp_oscal import oscal_generate_ssp

        mock_profile.return_value = _mock_fedramp_profile()
        mock_read.return_value = (
            "## AC-2 Account Management\n"
            "VirtualDojo manages accounts via Entra ID.\n"
            "## SI-4 Monitoring\n"
            "We use Cloud Monitoring for SI-4.\n"
        )

        result = oscal_generate_ssp.invoke({
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "with implementation descriptions from SSP markdown" in result

        committed_json = json.loads(mock_commit.call_args[0][1])
        impl_reqs = committed_json["system-security-plan"]["control-implementation"]["implemented-requirements"]
        ac2_req = next(r for r in impl_reqs if r["control-id"] == "ac-2")
        assert "Entra ID" in ac2_req["statements"][0]["description"]


# ---------------------------------------------------------------------------
# oscal_generate_poam
# ---------------------------------------------------------------------------


class TestGeneratePoam:
    """Tests for oscal_generate_poam — OSCAL POA&M generation."""

    def test_unauthorized(self):
        from tools.fedramp_oscal import oscal_generate_poam

        result = oscal_generate_poam.invoke({
            "conversation_id": CONV_ID,
            "user_email": UNAUTHORIZED_EMAIL,
        })
        assert "not authorized" in result

    @patch("tools.fedramp_oscal._commit_file", return_value="poamsha")
    @patch("tools.fedramp_oscal._read_github_file", return_value=None)
    def test_empty_poam_placeholder(self, mock_read, mock_commit):
        from tools.fedramp_oscal import oscal_generate_poam

        result = oscal_generate_poam.invoke({
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "OSCAL POA&M generated" in result
        assert "placeholder" in result

        committed_json = json.loads(mock_commit.call_args[0][1])
        poam = committed_json["plan-of-action-and-milestones"]
        assert poam["metadata"]["oscal-version"] == OSCAL_VERSION
        assert len(poam["poam-items"]) == 1
        assert poam["poam-items"][0]["status"] == "closed"

    @patch("tools.fedramp_oscal._commit_file", return_value="poamsha2")
    @patch("tools.fedramp_oscal._read_github_file")
    def test_parses_markdown_table(self, mock_read, mock_commit):
        from tools.fedramp_oscal import oscal_generate_poam

        mock_read.return_value = (
            "| Control ID | Description | Risk Level | Milestone | Due Date | Status |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| AC-2 | Account management gap | High | Implement IdP | 2026-06-01 | open |\n"
            "| SI-4 | Monitoring improvement | Moderate | Deploy SIEM | 2026-07-01 | open |\n"
        )

        result = oscal_generate_poam.invoke({
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "2 items" in result or "POA&M items: 2" in result
        assert "parsed" in result

        committed_json = json.loads(mock_commit.call_args[0][1])
        items = committed_json["plan-of-action-and-milestones"]["poam-items"]
        assert len(items) == 2
        assert "AC-2" in items[0]["title"]
        assert items[0]["associated-risks"][0]["risk-level"] == "high"


# ---------------------------------------------------------------------------
# oscal_generate_assessment_results
# ---------------------------------------------------------------------------


class TestGenerateAssessmentResults:
    """Tests for oscal_generate_assessment_results."""

    @patch("tools.fedramp_oscal.fedramp_collect_evidence", create=True)
    def test_assessment_structure(self, mock_evidence_tool):
        from tools.fedramp_oscal import oscal_generate_assessment_results

        # Mock the evidence tool's invoke method
        mock_evidence_tool.invoke = MagicMock(return_value=(
            "=== AC: Access Control === (project: test-project)\n"
            "[PASS] Service account inventory — 3 service account bindings found\n"
            "[FAIL] Overly broad roles — 1 binding with Owner/Editor\n"
        ))

        with patch("tools.fedramp_oscal.fedramp_collect_evidence", mock_evidence_tool):
            result = oscal_generate_assessment_results.invoke({
                "control_family": "AC",
                "project_id": "test-project",
                "conversation_id": CONV_ID,
                "user_email": AUTHORIZED_EMAIL,
            })

        assert "Assessment Results generated" in result
        assert "AC" in result
        # Pending upload should be stored
        assert CONV_ID in _pending_file_uploads
        content = json.loads(_pending_file_uploads[CONV_ID]["content"])
        ar = content["assessment-results"]
        assert ar["metadata"]["oscal-version"] == OSCAL_VERSION
        assert len(ar["results"]) == 1
        assert len(ar["results"][0]["observations"]) > 0


# ---------------------------------------------------------------------------
# oscal_migrate_from_markdown
# ---------------------------------------------------------------------------


class TestMigrateFromMarkdown:
    """Tests for oscal_migrate_from_markdown."""

    def test_unauthorized(self):
        from tools.fedramp_oscal import oscal_migrate_from_markdown

        result = oscal_migrate_from_markdown.invoke({
            "file_path": "test.md",
            "document_type": "ssp",
            "conversation_id": CONV_ID,
            "user_email": UNAUTHORIZED_EMAIL,
        })
        assert "not authorized" in result

    def test_invalid_document_type(self):
        from tools.fedramp_oscal import oscal_migrate_from_markdown

        result = oscal_migrate_from_markdown.invoke({
            "file_path": "test.md",
            "document_type": "invalid_type",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "Invalid document_type" in result
        assert "ssp" in result
        assert "poam" in result

    @patch("tools.fedramp_oscal._read_github_file")
    def test_file_not_found(self, mock_read):
        from tools.fedramp_oscal import oscal_migrate_from_markdown

        mock_read.return_value = None

        result = oscal_migrate_from_markdown.invoke({
            "file_path": "nonexistent.md",
            "document_type": "ssp",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "File not found" in result

    @patch("tools.fedramp_oscal._read_github_file")
    def test_migrate_ssp(self, mock_read):
        from tools.fedramp_oscal import oscal_migrate_from_markdown

        mock_read.return_value = (
            "## AC-2 Account Management\n"
            "VirtualDojo uses Entra ID.\n"
            "## SI-4 Monitoring\n"
            "Cloud Monitoring is configured.\n"
        )

        result = oscal_migrate_from_markdown.invoke({
            "file_path": "FedRAMP-Moderate-SSP.md",
            "document_type": "ssp",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "migration complete" in result.lower() or "Markdown migration complete" in result
        assert "2 control implementations" in result
        assert CONV_ID in _pending_file_uploads
        content = json.loads(_pending_file_uploads[CONV_ID]["content"])
        assert "system-security-plan" in content

    @patch("tools.fedramp_oscal._read_github_file")
    def test_migrate_poam(self, mock_read):
        from tools.fedramp_oscal import oscal_migrate_from_markdown

        mock_read.return_value = (
            "| Control ID | Description | Risk Level |\n"
            "| --- | --- | --- |\n"
            "| AC-2 | Account gap | High |\n"
        )

        result = oscal_migrate_from_markdown.invoke({
            "file_path": "poam.md",
            "document_type": "poam",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "1 POA&M items" in result
        content = json.loads(_pending_file_uploads[CONV_ID]["content"])
        assert "plan-of-action-and-milestones" in content

    @patch("tools.fedramp_oscal._read_github_file")
    def test_migrate_component(self, mock_read):
        from tools.fedramp_oscal import oscal_migrate_from_markdown

        mock_read.return_value = (
            "## Cloud Run\nServerless container runtime.\n"
            "## AlloyDB\nManaged PostgreSQL.\n"
        )

        result = oscal_migrate_from_markdown.invoke({
            "file_path": "components.md",
            "document_type": "component",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "2 component definitions" in result
        content = json.loads(_pending_file_uploads[CONV_ID]["content"])
        assert "component-definition" in content

    @patch("tools.fedramp_oscal._read_github_file")
    def test_migrate_policy(self, mock_read):
        from tools.fedramp_oscal import oscal_migrate_from_markdown

        mock_read.return_value = (
            "## Access Control (AC)\n"
            "VirtualDojo enforces least privilege.\n"
            "## System Integrity (SI)\n"
            "Integrity checks are performed daily.\n"
        )

        result = oscal_migrate_from_markdown.invoke({
            "file_path": "policy.md",
            "document_type": "policy",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "2 policy sections" in result
        content = json.loads(_pending_file_uploads[CONV_ID]["content"])
        assert "policy-document" in content


# ---------------------------------------------------------------------------
# oscal_update_control
# ---------------------------------------------------------------------------


class TestUpdateControl:
    """Tests for oscal_update_control — update an SSP control implementation."""

    def test_unauthorized(self):
        from tools.fedramp_oscal import oscal_update_control

        result = oscal_update_control.invoke({
            "control_id": "ac-2",
            "implementation_description": "new text",
            "conversation_id": CONV_ID,
            "user_email": UNAUTHORIZED_EMAIL,
        })
        assert "not authorized" in result

    @patch("tools.fedramp_oscal._read_github_json", return_value=None)
    def test_ssp_not_found(self, mock_read):
        from tools.fedramp_oscal import oscal_update_control

        result = oscal_update_control.invoke({
            "control_id": "ac-2",
            "implementation_description": "new text",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "SSP not found" in result

    @patch("tools.fedramp_oscal._read_github_json")
    def test_control_not_found_in_ssp(self, mock_read):
        from tools.fedramp_oscal import oscal_update_control

        mock_read.return_value = _mock_ssp_json()

        result = oscal_update_control.invoke({
            "control_id": "zz-99",
            "implementation_description": "new text",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "not found in the SSP" in result

    @patch("tools.fedramp_oscal._read_github_json")
    def test_update_shows_diff(self, mock_read):
        from tools.fedramp_oscal import oscal_update_control

        mock_read.return_value = _mock_ssp_json()

        result = oscal_update_control.invoke({
            "control_id": "ac-2",
            "implementation_description": "VirtualDojo now uses Entra ID with Conditional Access for AC-2.",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "AC-2" in result
        assert "Before:" in result
        assert "After:" in result
        assert "Entra ID" in result
        assert CONV_ID in _pending_file_uploads

        # Verify the updated content
        updated_ssp = json.loads(_pending_file_uploads[CONV_ID]["content"])
        ac2_req = next(
            r for r in updated_ssp["system-security-plan"]["control-implementation"]["implemented-requirements"]
            if r["control-id"] == "ac-2"
        )
        assert "Entra ID" in ac2_req["statements"][0]["description"]


# ---------------------------------------------------------------------------
# oscal_link_evidence
# ---------------------------------------------------------------------------


class TestLinkEvidence:
    """Tests for oscal_link_evidence — link evidence to assessment results."""

    def test_unauthorized(self):
        from tools.fedramp_oscal import oscal_link_evidence

        result = oscal_link_evidence.invoke({
            "control_id": "ac-2",
            "evidence_description": "IAM screenshot",
            "evidence_url": "https://evidence.example.com/screenshot.png",
            "conversation_id": CONV_ID,
            "user_email": UNAUTHORIZED_EMAIL,
        })
        assert "not authorized" in result

    @patch("tools.fedramp_oscal._read_github_json", return_value=None)
    def test_creates_new_ar_when_missing(self, mock_read):
        from tools.fedramp_oscal import oscal_link_evidence

        result = oscal_link_evidence.invoke({
            "control_id": "ac-2",
            "evidence_description": "IAM audit screenshot",
            "evidence_url": "https://evidence.example.com/iam.png",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "Evidence linked" in result
        assert "AC-2" in result
        assert CONV_ID in _pending_file_uploads

        content = json.loads(_pending_file_uploads[CONV_ID]["content"])
        assert "assessment-results" in content
        observations = content["assessment-results"]["results"][0]["observations"]
        assert len(observations) == 1
        assert observations[0]["relevant-evidence"][0]["href"] == "https://evidence.example.com/iam.png"


# ---------------------------------------------------------------------------
# oscal_validate_package
# ---------------------------------------------------------------------------


class TestValidatePackage:
    """Tests for oscal_validate_package — structural validation."""

    @patch("tools.fedramp_oscal._read_github_json", return_value=None)
    def test_file_not_found(self, mock_read):
        from tools.fedramp_oscal import oscal_validate_package

        result = oscal_validate_package.invoke({"file_path": "oscal/ssp.json"})
        assert "File not found" in result

    @patch("tools.fedramp_oscal._read_github_json")
    def test_valid_ssp_passes(self, mock_read):
        from tools.fedramp_oscal import oscal_validate_package

        mock_read.return_value = _mock_ssp_json()

        result = oscal_validate_package.invoke({
            "file_path": "oscal/system-security-plan.json",
        })
        assert "SSP" in result
        assert "PASS" in result
        # Check specific validations
        assert "Root UUID present" in result
        assert "Metadata present" in result
        assert "oscal-version matches" in result

    @patch("tools.fedramp_oscal._read_github_json")
    def test_detects_missing_fields(self, mock_read):
        from tools.fedramp_oscal import oscal_validate_package

        # Minimal SSP missing required fields
        mock_read.return_value = {
            "system-security-plan": {
                "uuid": "not-a-valid-uuid",
                "metadata": {"title": "Test"},
            }
        }

        result = oscal_validate_package.invoke({
            "file_path": "oscal/system-security-plan.json",
        })
        assert "FAIL" in result
        # Should detect missing fields
        assert "system-characteristics" in result or "control-implementation" in result

    @patch("tools.fedramp_oscal._read_github_json")
    def test_validates_uuid_format(self, mock_read):
        from tools.fedramp_oscal import oscal_validate_package

        mock_read.return_value = {
            "system-security-plan": {
                "uuid": "bad-uuid-format",
                "metadata": {
                    "title": "Test",
                    "last-modified": "2026-04-01",
                    "version": "1.0",
                    "oscal-version": OSCAL_VERSION,
                },
            }
        }

        result = oscal_validate_package.invoke({
            "file_path": "oscal/system-security-plan.json",
        })
        assert "[FAIL] Root UUID format valid" in result

    @patch("tools.fedramp_oscal._read_github_json")
    def test_validates_poam_structure(self, mock_read):
        from tools.fedramp_oscal import oscal_validate_package

        mock_read.return_value = {
            "plan-of-action-and-milestones": {
                "uuid": str(uuid.uuid4()),
                "metadata": {
                    "title": "Test POA&M",
                    "last-modified": "2026-04-01",
                    "version": "1.0",
                    "oscal-version": OSCAL_VERSION,
                },
                "import-ssp": {"href": "system-security-plan.json"},
                "poam-items": [
                    {
                        "uuid": str(uuid.uuid4()),
                        "title": "Item 1",
                        "description": "Test item",
                    }
                ],
            }
        }

        result = oscal_validate_package.invoke({"file_path": "oscal/poam.json"})
        assert "POA&M" in result
        assert "poam-items present" in result
        assert "import-ssp present" in result

    @patch("tools.fedramp_oscal._read_github_json")
    def test_unknown_document_type(self, mock_read):
        from tools.fedramp_oscal import oscal_validate_package

        mock_read.return_value = {"some-other-root": {"uuid": "abc"}}

        result = oscal_validate_package.invoke({"file_path": "oscal/unknown.json"})
        assert "unknown" in result.lower()
        assert "FAIL" in result

    @patch("tools.fedramp_oscal._read_github_json")
    def test_wrong_oscal_version(self, mock_read):
        from tools.fedramp_oscal import oscal_validate_package

        ssp = _mock_ssp_json()
        ssp["system-security-plan"]["metadata"]["oscal-version"] = "0.9.0"
        mock_read.return_value = ssp

        result = oscal_validate_package.invoke({
            "file_path": "oscal/system-security-plan.json",
        })
        assert "found '0.9.0'" in result
        assert f"expected '{OSCAL_VERSION}'" in result


# ---------------------------------------------------------------------------
# oscal_catalog_lookup
# ---------------------------------------------------------------------------


class TestCatalogLookup:
    """Tests for oscal_catalog_lookup — NIST 800-53 control lookup."""

    @patch("tools.fedramp_oscal._get_fedramp_profile")
    @patch("tools.fedramp_oscal._get_nist_catalog")
    def test_find_control(self, mock_catalog, mock_profile):
        from tools.fedramp_oscal import oscal_catalog_lookup

        mock_catalog.return_value = _mock_nist_catalog()
        mock_profile.return_value = _mock_fedramp_profile()

        result = oscal_catalog_lookup.invoke({"control_id": "AC-2"})
        assert "AC-2" in result
        assert "Account Management" in result
        assert "organization manages information system accounts" in result
        assert "FedRAMP Moderate" in result

    @patch("tools.fedramp_oscal._get_nist_catalog")
    def test_control_not_found(self, mock_catalog):
        from tools.fedramp_oscal import oscal_catalog_lookup

        mock_catalog.return_value = _mock_nist_catalog()

        result = oscal_catalog_lookup.invoke({"control_id": "ZZ-99"})
        assert "not found" in result
        assert "ZZ-99" in result

    @patch("tools.fedramp_oscal._get_fedramp_profile")
    @patch("tools.fedramp_oscal._get_nist_catalog")
    def test_lookup_case_insensitive(self, mock_catalog, mock_profile):
        from tools.fedramp_oscal import oscal_catalog_lookup

        mock_catalog.return_value = _mock_nist_catalog()
        mock_profile.return_value = _mock_fedramp_profile()

        result = oscal_catalog_lookup.invoke({"control_id": "ac-2"})
        assert "Account Management" in result

    @patch("tools.fedramp_oscal._get_fedramp_profile")
    @patch("tools.fedramp_oscal._get_nist_catalog")
    def test_shows_parameters(self, mock_catalog, mock_profile):
        from tools.fedramp_oscal import oscal_catalog_lookup

        mock_catalog.return_value = _mock_nist_catalog()
        mock_profile.return_value = _mock_fedramp_profile()

        result = oscal_catalog_lookup.invoke({"control_id": "AC-2"})
        assert "Parameters" in result
        assert "ac-2_prm_1" in result

    @patch("tools.fedramp_oscal._get_fedramp_profile")
    @patch("tools.fedramp_oscal._get_nist_catalog")
    def test_shows_enhancements(self, mock_catalog, mock_profile):
        from tools.fedramp_oscal import oscal_catalog_lookup

        mock_catalog.return_value = _mock_nist_catalog()
        mock_profile.return_value = _mock_fedramp_profile()

        result = oscal_catalog_lookup.invoke({"control_id": "AC-2"})
        assert "Enhancements" in result
        assert "Automated System Account Management" in result


# ---------------------------------------------------------------------------
# oscal_render_pdf
# ---------------------------------------------------------------------------


class TestRenderPdf:
    """Tests for oscal_render_pdf — PDF generation from OSCAL documents."""

    def test_unauthorized(self):
        from tools.fedramp_oscal import oscal_render_pdf

        result = oscal_render_pdf.invoke({
            "document_type": "ssp",
            "conversation_id": CONV_ID,
            "user_email": UNAUTHORIZED_EMAIL,
        })
        assert "not authorized" in result

    def test_invalid_document_type(self):
        from tools.fedramp_oscal import oscal_render_pdf

        result = oscal_render_pdf.invoke({
            "document_type": "invalid",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "Invalid document_type" in result

    @patch("tools.fedramp_oscal._read_github_json", return_value=None)
    def test_document_not_found(self, mock_read):
        from tools.fedramp_oscal import oscal_render_pdf

        result = oscal_render_pdf.invoke({
            "document_type": "ssp",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "not found" in result

    @patch("tools.fedramp_oscal._read_github_json")
    def test_ssp_pdf_generation(self, mock_read):
        from tools.fedramp_oscal import oscal_render_pdf

        mock_read.return_value = _mock_ssp_json()

        result = oscal_render_pdf.invoke({
            "document_type": "ssp",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "PDF generated" in result
        assert "System Security Plan" in result
        assert "Pages:" in result
        assert CONV_ID in _pending_file_uploads

        # Verify the pending upload contains valid base64 PDF data
        pending = _pending_file_uploads[CONV_ID]
        assert pending["is_binary"] is True
        assert pending["content_type"] == "application/pdf"
        # Decode base64 to verify it is valid
        pdf_bytes = base64.b64decode(pending["content"])
        assert len(pdf_bytes) > 0
        # PDF files start with %PDF
        assert pdf_bytes[:4] == b"%PDF"

    @patch("tools.fedramp_oscal._read_github_json")
    def test_poam_pdf_generation(self, mock_read):
        from tools.fedramp_oscal import oscal_render_pdf

        mock_read.return_value = {
            "plan-of-action-and-milestones": {
                "uuid": str(uuid.uuid4()),
                "metadata": {
                    "title": "Test POA&M",
                    "last-modified": "2026-04-01",
                    "version": "1.0",
                    "oscal-version": OSCAL_VERSION,
                },
                "import-ssp": {"href": "system-security-plan.json"},
                "poam-items": [
                    {
                        "uuid": str(uuid.uuid4()),
                        "title": "POA&M Item: AC-2",
                        "description": "Account management gap",
                        "status": "open",
                        "associated-risks": [
                            {"uuid": str(uuid.uuid4()), "title": "Risk", "risk-level": "high"}
                        ],
                        "milestones": [
                            {"uuid": str(uuid.uuid4()), "title": "Implement IdP"}
                        ],
                    }
                ],
            }
        }

        result = oscal_render_pdf.invoke({
            "document_type": "poam",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        })
        assert "PDF generated" in result
        assert "POA&M" in result
        pending = _pending_file_uploads[CONV_ID]
        pdf_bytes = base64.b64decode(pending["content"])
        assert pdf_bytes[:4] == b"%PDF"
