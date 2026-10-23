import re
from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/unique_statements.csv")
OUTPUT_PATH = Path("data/processed/unique_statements_cleaned.csv")

MOJIBAKE_FIXES = {
    "â€™": "’", "â€œ": "“", "â€\x9d": "”", "â€“": "–", "â€”": "—",
    "â€˜": "‘", "Â®": "®", "â„¢": "™", "Â": "",
}

NOISE_PATTERNS = [
    r"\blift\b.*\blbs?\b", r"\bstand\b.*\bhours?\b", r"\bwalk\b.*\bhours?\b",
    r"401\s?k", r"paid vacation", r"health savings", r"employee discounts?",
    r"parental leave", r"short.?term disability", r"long.?term disability",
    r"life insurance", r"^dental$", r"vision insurance", r"tuition assistance",
    r"commuter benefits", r"pet.?insurance", r"discounted rent", r"paid sick time",
    r"wellness bonus", r"paid holidays", r"^salary ranges?$", r"discretionary incentive",
    r"shift differential", r"hiring rate", r"displayed rate", r"^market rate$",
    r"background investigation", r"arrest and/or conviction", r"fair chance ordinance",
    r"equal employment opportunity", r"motor vehicle report", r"education verification",
    r"social security verification", r"drug testing", r"pre-employment", r"tuberculosis",
    r"background check", r"public trust clearance", r"^selective service$",
    r"drug and alcohol testing records", r"background/security investigation",
    r"^job id$", r"closing date", r"application deadline", r"posting location",
    r"^start date$", r"anticipated graduation", r"apply.{0,3}button",
    r"application status link", r"uploaded documents", r"scanned copies",
    r"^cover letter$", r"^resume/cv", r"business address", r"e-mail address",
    r"qualification determinations", r"application question", r"details page",
    r"information link", r"left hand navigation menu",
    r"\bgs-\d", r"\be-\d\b", r"\bo-\d\b", r"\(\s*$", r"\.\(?:\d",
    r"\b(?:days|development|experience|position|years|fields|construction)\d",
    r"varying weather conditions", r"uneven ground", r"visual strain",
    r"noxious odors", r"toxic or caustic chemicals", r"electrical shock",
    r"bright/dim lights", r"^mechanical parts$",
    r"relocation assistance", r"travel up to \d", r"commute/relocate",
]
NOISE_RE = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)

KNOWN_REAL_BARE_WORDS = {
    "shg", "sfg", "communications", "marketing", "manufacturing", "macola", "indesign",
    "cinema4d", "blender", "powerpoint", "figma", "decision-making", "construction",
    "management", "word", "zoning", "execution", "patent", "uspto", "wipo", "efs-web",
    "tsdr", "regression", "clustering", "maxdiff", "conjoint", "qualtrics", "aytm",
    "ppe", "networking", "security", "compliance", "budgeting", "team-oriented",
    "normalization", "talend", "xml/xslt", "cross-training", "sales", "multi-task",
    "c#", "c++", "documentation", "osha", "hhs", "e-discovery", "litigation",
    "permitting", "self-discipline", "rnc-ob", "blueprints", "sharepoint", "finance",
    "engineering", "iso9001", "iso13485", "gdps", "teaching", "planning", "html",
    "css", "annotation", "miro", "camtasia", "usability", "ada", "roi", "learning",
    "leadership", "cots", "english", "micros", "toast", "solidworks", "revit",
    "advocacy", "outlook", "maintenance", "cdl", "gc/ms", "adme", "sops", "nursery",
    "driving", "forklift", "statistics", "mathematics", "nlp", "hql", "dbscan",
    "mba", "enovia", "operations", "biotech", "crf", "sketch", "design", "reasoning",
    "presentation", "cybersecurity", "nist", "linux", "windows", "gcp",
    "infrastructure", "jenkins", "ansible", "terraform", "gmail", "devsecops",
    "vmware", "slack", "servicenow", "seo", "sem", "emailmarketing", "cms",
    "research", "evaluation", "monitoring", "validation", "rcts", "instrumentation",
    "p&l", "aoac", "usp", "iso", "humanities", "stem", "ieps", "data-management",
    "notion", "government", "sketchup", "advertising", "direct-mail", "production",
    "ui/ux", "xd", "storyboards", "ux", "cad", "cam", "plc", "bottling", "packaging",
    "architecture", "agile", "symphony", "ai", "retail", "zoom", "recruitment",
    "linkedin", "css3", "html5", "adobe", "ceqa", "nepa", "investing", "analysis",
    "spac", "animation", "medicaid", "ltss", "msde", "aca", "landscaping", "arrt",
    "rn", "rcis", "rces", "itsm", "legislation", "vfd", "welding", "powershell",
    "hyper-v", "puppet", "hardware", "synthesis", "polymerization", "urethanes",
    "silicones", "epoxies", "r&d", "airflow", "dbt", "dashboarding", "analytics",
    "econometric", "supervision", "estimating", "scheduling", "hospitality", "opera",
    "fidelio", "pedagogy", "communication", "long-haul", "buffer", "hootsuite",
    "pricing", "cx", "insurance", "implementation", "configuration",
    "troubleshooting", "sms", "electronics", "nec", "bio-hazard", "xml", "ziosk",
    "scikit-learn", "tensorflow", "pytorch", "rstudio", "customer-centric",
    "devops", "flosum", "cns", "g-suite", "plumbing", "hvac", "iri", "writing",
    "reporting", "ifrs", "sec", "tsx", "cpa", "cataloging", "ils", "copyright",
    "acquisitions", "nutrition", "car-sharing", "saas", "fundraising",
    "customization", "crm", "epic", "autocad", "onenote", "cpff", "t&m", "i-9",
    "hris", "b2b", "finra", "healthcare", "s3", "sqs", "sns", "confluence",
    "consulting", "veeva", "sas", "sanitation", "safety", "wawf", "piee",
    "ipc-a-610", "dcma", "accounting", "sigma", "lean", "word-processing",
    "spreadsheets", "mirroring", "replication", "workday", "phenom", "easements",
    "liability", "cpm", "rpa", "persuasion", "ml/gen-ai", "react", "nodejs",
    "typescript", "next.js", "centos", "aurora", "gen-ai", "no-sql", "msk",
    "journalism", "storytelling", "visio", "front-end", "nextjs", "optimization",
    "testing", "mining", "lisp", "regex", "tractor-trailer", "sustainability",
    "curriculum", "as9100", "spring", "microservices", "angular", "hibernate",
    "maven", "junit", "apis", "algorithms", "kafka", "ignite", "redis", "rmm",
    "solarwinds", "cpr", "cdc", "domo", "iis", "depositions", "visualization",
    "scripting", "containerization", "mesos", "self-motivation", "fabrication",
    "icu", "nursing", "acls", "ngo", "chemistry", "microbiology", "bloomberg",
    "storage", "bash", "migration", "hplc", "pharma/biotech", "hr", "rcra", "dot",
    "dea", "pharmaceutical", "salesforce.com", "warehousing/distribution",
    "outreach", "8d/dmaic", "investigations", "claims", "auctions", "cpg", "ift",
    "rca", "pid", "microstation", "mdm", "reading/debugging", "uat", "printing",
    "copywriting", "accountability", "eq", "email", "hemodynamics", "bls",
    "macintosh", "banking", "tracking", "forecasting", "itar",
    "sap", "python", "excel", "java", "sql", "azure", "aws", "tableau",
    "salesforce", "jira", "photoshop", "illustrator", "docker", "kubernetes",
    "snowflake", "fda", "french", "spanish", "mandarin", "javascript",
}


def fix_mojibake(text):
    for bad, good in MOJIBAKE_FIXES.items():
        text = text.replace(bad, good)
    return text


def main():
    df = pd.read_csv(INPUT_PATH)
    df["statement"] = df["statement"].apply(fix_mojibake)

    before = len(df)
    noisy_mask = df["statement"].str.contains(NOISE_RE, na=False, regex=True)

    is_single_word = df["statement"].str.split().str.len() == 1
    is_unconfirmed_bare_word = is_single_word & ~df["statement"].str.lower().isin(KNOWN_REAL_BARE_WORDS)

    remove_mask = noisy_mask | is_unconfirmed_bare_word
    cleaned = df[~remove_mask].reset_index(drop=True)

    cleaned.to_csv(OUTPUT_PATH, index=False)

    print(f"Loaded {before} statements")
    print(f"Removed {noisy_mask.sum()} matching confirmed noise patterns")
    print(f"Removed {is_unconfirmed_bare_word.sum()} bare words not on the reviewed keep-list")
    print(f"Kept {len(cleaned)}, saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()