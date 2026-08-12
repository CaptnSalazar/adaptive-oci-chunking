# Access Control Policy

This policy describes authentication, authorization, support access, and audit
requirements for internal systems. It is intentionally structured like a small
handbook so chunking strategies have meaningful section boundaries to preserve.

## Authentication

All employees must authenticate with single sign-on before accessing production
systems. Multi-factor authentication is required for administrative consoles,
customer data tools, billing systems, analytics warehouses, and deployment
pipelines. See Section 2.1 for exception handling.

### Passwordless access

Teams should prefer passkeys or hardware-backed authentication for high-risk
systems. Shared passwords are not allowed. Recovery codes must be stored in an
approved vault and rotated after use.

### Temporary exceptions

Temporary exceptions may be granted for incident response, migration windows, or
vendor outages. Each exception must include an owner, expiration date, risk
acceptance note, and rollback plan.

## Authorization

Access is granted by role and reviewed quarterly. Production administrator
groups must remain separate from development administrator groups. Privileged
access should be time-bound whenever the platform supports it.

### Least privilege

Managers and service owners are responsible for confirming that each user has
only the permissions needed for current work. Inactive users must be removed
within seven days of role change or termination.

### Service accounts

Service accounts require named owners, documented purpose, and rotation policy.
Long-lived credentials must be avoided. If long-lived credentials are required,
the owner must document why short-lived credentials are not sufficient.

## Support access

Support access to customer environments is limited to approved tickets. The
ticket must identify the customer, the support engineer, the reason for access,
and the expected end time. Emergency access must be reviewed after the incident.

## Audit logging

Authentication events, privilege changes, support sessions, and failed access
attempts must be logged. Logs must include user identity, source system, target
resource, timestamp, and outcome. Retention must follow the compliance schedule.

## Review cadence

The security team reviews this policy every six months. Product teams may
request clarifications through the governance queue. Material changes require
approval from security, legal, and the affected service owners.

## Data access workflow

Requests for customer data access must begin with a documented business reason.
The requester must identify the product area, the customer account, the data
category, and the expected duration of access. The approver must be outside the
requester's reporting chain for high-risk systems.

### Approval evidence

Approval evidence must remain attached to the ticket. Screenshots, chat
messages, or verbal approvals are not sufficient unless they are copied into the
ticket with timestamp, approver identity, and expiration. The access platform
should block requests that do not include complete evidence.

### Revocation

Access must be revoked automatically at expiration. If automatic revocation is
not available, the owner must schedule manual revocation and confirm completion
in the ticket. Missed revocations are treated as policy exceptions and reviewed
by the security team.

## Vendor access

Vendors may access internal systems only through approved federation or managed
accounts. Personal email accounts are not allowed. Vendor permissions must be
limited to the contracted service scope and reviewed at least monthly.

### Emergency vendor access

Emergency vendor access requires incident commander approval. The incident
commander must document the start time, intended work, systems touched, and
expected end time. After the incident, the service owner must review logs and
confirm that the vendor performed only authorized actions.

## Monitoring and alerting

Security monitoring must alert on privilege escalation, repeated failed login
attempts, access from unexpected geographies, and administrative actions outside
approved windows. Alerts must include enough context for responders to decide
whether the activity is expected.

### Alert review

Alert rules must be reviewed quarterly. Noisy alerts should be tuned rather than
ignored. Disabled alerts require an owner, reason, expiration, and compensating
control. Permanent alert removals require security approval.
