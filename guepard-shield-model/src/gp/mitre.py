"""MITRE ATT&CK mapping for LID-DS scenarios.

Maps each scenario name to one or more MITRE ATT&CK technique IDs.
Used by C7a (coverage matrix) in P4.

Reference: https://attack.mitre.org/techniques/
"""

# fmt: off
SCENARIO_TO_TECHNIQUES: dict[str, list[str]] = {
    # LID-DS-2019 and LID-DS-2021 shared scenarios
    "Bruteforce_CWE-307":   ["T1110"],            # Brute Force
    "CVE-2012-2122":        ["T1190", "T1078"],   # Exploit Public-Facing App, Valid Accounts
    "CVE-2014-0160":        ["T1190", "T1552"],   # Heartbleed → Exploit + Unsecured Credentials
    "CVE-2017-7529":        ["T1190"],            # Nginx integer overflow
    "CVE-2018-3760":        ["T1190", "T1083"],   # Sprockets path traversal → Exploit + File Discovery
    "CVE-2019-5418":        ["T1190", "T1083"],   # Rails file disclosure
    "EPS_CWE-434":          ["T1190", "T1505.003"],  # Unrestricted file upload → Web Shell
    "PHP_CWE-434":          ["T1190", "T1505.003"],  # PHP file upload
    "SQL_Injection_CWE-89": ["T1190"],            # SQL injection
    "ZipSlip":              ["T1190", "T1083"],   # Zip path traversal

    # LID-DS-2021 only
    "CVE-2017-12635_6":     ["T1068", "T1190"],   # CouchDB privilege escalation
    "CVE-2020-13942":       ["T1190", "T1059"],   # Apache Unomi RCE → Exploit + Command Execution
    "CVE-2020-23839":       ["T1190"],            # GetSimpleCMS info disclosure
    "CVE-2020-9484":        ["T1190", "T1059"],   # Tomcat session deserialization → RCE
    "CWE-89-SQL-injection": ["T1190"],            # SQL injection (2021 variant)
    "Juice-Shop":           ["T1190", "T1059"],   # OWASP Juice Shop multi-vuln
}

TECHNIQUE_NAMES: dict[str, str] = {
    "T1059":     "Command and Scripting Interpreter",
    "T1068":     "Exploitation for Privilege Escalation",
    "T1078":     "Valid Accounts",
    "T1083":     "File and Directory Discovery",
    "T1110":     "Brute Force",
    "T1190":     "Exploit Public-Facing Application",
    "T1505.003": "Server Software Component: Web Shell",
    "T1552":     "Unsecured Credentials",
}
# fmt: on


def techniques_for(scenario: str) -> list[str]:
    """Return MITRE technique IDs for a given scenario name."""
    return SCENARIO_TO_TECHNIQUES.get(scenario, [])


def scenarios_for(technique_id: str) -> list[str]:
    """Return all scenario names associated with a given technique ID."""
    return [s for s, ts in SCENARIO_TO_TECHNIQUES.items() if technique_id in ts]
