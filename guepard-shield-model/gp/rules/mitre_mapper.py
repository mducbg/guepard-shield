"""Map LID-DS-2021 scenarios to MITRE ATT&CK techniques."""

from __future__ import annotations

from typing import Dict, List
from collections import defaultdict


class LIDDS2021MITREMapper:
    """Maps LID-DS-2021 scenario names to MITRE ATT&CK technique IDs."""

    SCENARIO_TO_MITRE: Dict[str, List[str]] = {
        "Bruteforce_CWE-307": ["T1110"],
        "SQL_Injection":       ["T1190"],
        "Command_Injection":   ["T1059", "T1190"],
        "LFI":                 ["T1190", "T1083"],
        "RFI":                 ["T1190", "T1105"],
        "Path_Traversal":      ["T1083", "T1190"],
        "Auth_Bypass":         ["T1078", "T1098"],
        "CVE-2012-2122":       ["T1190", "T1078"],
        "CVE-2014-0160":       ["T1190", "T1040"],
        "CVE-2017-7529":       ["T1190"],
        "CVE-2017-12635":      ["T1190", "T1078"],
        "CVE-2017-1000112":    ["T1190", "T1068"],
        "CVE-2018-3760":       ["T1190"],
        "CVE-2019-5418":       ["T1190"],
        "CVE-2019-5419":       ["T1190"],
        "CVE-2019-5420":       ["T1190"],
        "Reverse_Shell":       ["T1059", "T1071"],
        "Backdoor":            ["T1059", "T1100"],
        "PrivEsc":             ["T1068", "T1078"],
        "Exfiltration":        ["T1041", "T1048"],
        "DoS":                 ["T1498", "T1499"],
    }

    MITRE_DESCRIPTIONS: Dict[str, str] = {
        "T1110": "Brute Force",
        "T1190": "Exploit Public-Facing Application",
        "T1059": "Command and Scripting Interpreter",
        "T1078": "Valid Accounts",
        "T1068": "Exploitation for Privilege Escalation",
        "T1040": "Network Sniffing",
        "T1083": "File and Directory Discovery",
        "T1098": "Account Manipulation",
        "T1100": "Web Shell",
        "T1105": "Ingress Tool Transfer",
        "T1071": "Application Layer Protocol",
        "T1041": "Exfiltration Over C2 Channel",
        "T1048": "Exfiltration Over Alternative Protocol",
        "T1498": "Network Denial of Service",
        "T1499": "Endpoint Denial of Service",
    }

    def map_scenario(self, scenario: str) -> List[str]:
        for prefix, techniques in self.SCENARIO_TO_MITRE.items():
            if scenario.startswith(prefix):
                return techniques
        return ["T1190"]

    def map_recording(self, recording_name: str) -> List[str]:
        parts = recording_name.split("_")
        for n in [3, 2, 1]:
            if len(parts) >= n:
                scenario = "_".join(parts[:n])
                techniques = self.map_scenario(scenario)
                if techniques != ["T1190"] or n == 1:
                    return techniques
        return ["T1190"]

    def analyze_rule_coverage(self, rule_idx: int, fired_recordings: List[str]) -> Dict:
        technique_counts: Dict[str, int] = defaultdict(int)
        scenario_counts: Dict[str, int] = defaultdict(int)

        for rec in fired_recordings:
            scenario = rec.split("_")[0]
            scenario_counts[scenario] += 1
            for t in self.map_recording(rec):
                technique_counts[t] += 1

        return {
            "rule_idx": rule_idx,
            "fired_on": len(fired_recordings),
            "top_scenarios": dict(sorted(scenario_counts.items(), key=lambda x: -x[1])[:5]),
            "mitre_techniques": {
                t: {"count": c, "description": self.MITRE_DESCRIPTIONS.get(t, "Unknown")}
                for t, c in sorted(technique_counts.items(), key=lambda x: -x[1])
            },
        }
