# Patient Data Handling and Consent Standard

# MERIDIAN HEALTH PARTNERS
## Data Handling Standard

**Document ID:** MHP-POL-DH-004
**Version:** 3.2
**Effective date:** 1 March 2025
**Owner:** Operations Policy Lead
**Approving officers:** Marguerite Osei-Lindqvist (Data Protection Officer); Dr. Tobias Renquist (Chief Medical Officer)
**Next scheduled review:** 1 March 2026
**Supersedes:** MHP-POL-DH-004 v3.1 (effective 1 September 2023)
**Applies to:** All 14 Meridian sites, the central Privacy Office, and all workforce members and contracted processors handling patient data

---

## 1. PURPOSE AND SCOPE

1.1 This Standard sets the minimum requirements for how Meridian Health Partners ("Meridian") collects consent for, accesses, retains, discloses, and responds to requests about patient personal data, and how it responds to breaches of that data.

1.2 This Standard covers data in any form: the MeridianChart electronic health record, paper charts, imaging archives, laboratory interfaces, billing systems, telehealth recordings, call-centre recordings, patient portal messages, and removable media.

1.3 Meridian operates under two regimes: the Digital Personal Data Protection Act 2023 (India) ("DPDP Act") and the Health Insurance Portability and Accountability Act (United States) ("HIPAA"). Sites 1 to 9 are subject to the DPDP Act. Sites 10 to 14 and the central billing centre are subject to HIPAA. The Privacy Office, MeridianChart, and all shared services are subject to both.

1.4 This Standard does not restate either regime. Where a requirement of the DPDP Act or HIPAA is stricter than a clause below for a given record, the stricter requirement governs, and the Privacy Office will issue a written site advisory within 10 business days of identifying the conflict.

1.5 The deadlines in this Standard are internal operating deadlines. They are set deliberately tighter than the outer limits Meridian understands the two regimes to impose, so that a missed internal deadline does not automatically become a missed legal one.

---

## 2. DEFINITIONS

2.1 **Patient data**: any information relating to an identified or identifiable patient, including demographic, clinical, financial, device, and communication data.

2.2 **Restricted category data**: records relating to mental health, substance use treatment, HIV status, reproductive and sexual health, genetic testing, and any patient under 18 receiving confidential adolescent services.

2.3 **Workforce member**: any employee, locum, trainee, volunteer, or agency worker with a MeridianChart or network credential.

2.4 **Processor**: any third party handling patient data on Meridian's behalf under a written agreement, including Halvorsen Cloud Systems (hosting), Pellucid Diagnostics (reference laboratory), and Tamarind Billing Services (claims).

2.5 **Awareness**: the moment any workforce member has a reasonable basis to believe a breach has occurred. It is not the moment the breach is confirmed.

2.6 **Business day**: Monday to Friday, excluding the public holidays listed in the annual Privacy Office calendar.

---

## 3. CONSENT CAPTURE

3.1 **When consent is captured.** Consent must be captured at first registration, and re-captured when the purpose, category of data, or recipient changes materially, and in any case at the first encounter after each 36-month anniversary of the last consent.

3.2 **Separate consents.** The following consents are captured as separate, unbundled items in ConsentVault. Refusal of any item other than 3.2(a) must not affect the patient's access to care.

| Ref | Consent item | Required for care? |
|---|---|---|
| 3.2(a) | Treatment and essential operations (scheduling, billing, clinical record) | Yes |
| 3.2(b) | Sharing with referring or receiving providers outside Meridian | No |
| 3.2(c) | Appointment and result reminders by SMS, email, or app notification | No |
| 3.2(d) | Use of de-identified data for quality improvement and audit | No |
| 3.2(e) | Research contact (being approached about studies) | No |
| 3.2(f) | Marketing and educational communications | No |
| 3.2(g) | Recording of telehealth sessions | No |

3.3 **Language and format.** The consent notice must be offered in English and in the patient's preferred language from the approved list of 11 languages. Reading level must not exceed Grade 8. If the patient cannot read the notice, front-desk staff must read it aloud and record "assisted" in ConsentVault together with the staff member's ID.

3.4 **Form of capture.** Consent must be recorded by one of: electronic signature on the registration tablet, signed paper form scanned within 1 business day, or recorded verbal consent for telephone registration with a second-staff witness ID. Pre-ticked boxes and default-on settings are prohibited.

3.5 **Minors and incapacity.**
(a) For patients under 18, consent is given by a parent or legal guardian, whose identity and relationship must be recorded.
(b) At the 18th birthday, MeridianChart automatically flags the record and consent must be re-captured from the patient at the next encounter. Until then, the guardian's consents stand, except that guardian portal access is suspended on the 18th birthday.
(c) For patients lacking capacity, consent is given by the legally authorised representative, with documentation attached to the record within 2 business days.
(d) Confidential adolescent services (clinic list in Annex B) are exempt from 3.5(a) in cases where the treating clinician documents that the patient is competent to consent. In such cases the guardian must not be given access to the related entries.

3.6 **Withdrawal.** A patient may withdraw any consent in 3.2(b) to 3.2(g) at any time by any front desk, by the portal, or by calling the Privacy Office line.
(a) Withdrawal must be recorded in ConsentVault within 1 business day.
(b) Marketing and reminder suppression must take effect within 2 business days.
(c) Withdrawal does not affect processing already carried out, and does not require deletion of the clinical record, which remains subject to Section 6.

3.7 **Emergency treatment.** Where a patient cannot give consent and treatment is urgent, care proceeds and the treating clinician must record the circumstances in the chart within 4 hours. Consent must be sought at the earliest practicable point after that.

3.8 **Evidence.** Each consent record must store: patient ID, item reference, version of the notice shown, date and time, capture method, staff ID, and language. Consent records must not be edited. Corrections are made by superseding entries.

3.9 **Audit.** Each site manager must sample 30 randomly selected new registrations per month and confirm completeness of consent records. Sites with an error rate above 3% must submit a corrective action plan to the Privacy Office within 10 business days.

---

## 4. MINIMUM NECESSARY ACCESS

4.1 **Principle.** Workforce members may access only the patient data needed to perform their current assigned duties for the patients assigned to or reasonably connected with those duties.

4.2 **Role tiers.** Access in MeridianChart is granted by role. No individual grants are permitted except under 4.4.

| Tier | Roles | Default scope |
|---|---|---|
| T1 | Front desk, schedulers | Demographics, appointments, insurance, consent status. No clinical notes. |
| T2 | Nurses, medical assistants | Full chart for patients at their own site and active encounters. No restricted category data unless on the treating team. |
| T3 | Physicians, advanced practice clinicians | Full chart for patients on their panel or with a documented consult or referral. |
| T4 | Billing and coding | Billing-relevant fields and diagnosis and procedure codes. No free-text notes. |
| T5 | Quality and audit analysts | De-identified or pseudonymised extracts only, unless approved under 4.8. |
| T6 | IT administrators | Infrastructure access only. No routine chart access. |

4.3 **Restricted category data.** Restricted category data is visible only to members of the treating team named on the care plan. Other T2 and T3 users see a masked entry reading "Restricted record" and cannot proceed without a break-glass event under 4.5.

4.4 **Exceptions.** An access grant outside a standard role tier requires a written request from the department head, approval from the Privacy Office, and an expiry date no later than 90 days from approval. Renewals require fresh approval.

4.5 **Break-glass access.**
(a) Any T2 or T3 user may override a chart restriction for a patient where delay would risk harm, by entering a reason of at least 20 characters.
(b) Each break-glass event generates an alert to the site manager and the Privacy Office immediately.
(c) The Privacy Office must review every break-glass event within 24 hours of the event, and 72 hours if the event falls on a weekend or holiday.
(d) Events without a justified clinical reason are treated as potential breaches under Section 8.

4.6 **Provisioning and removal.**
(a) New-starter access is granted only after completion of privacy training (4.10) and a signed confidentiality undertaking.
(b) Access must be removed within 4 hours of notified termination or resignation with immediate effect, and by 17:00 on the last working day otherwise.
(c) Accounts inactive for 60 consecutive days are suspended automatically. Accounts inactive for 180 days are deleted.
(d) Transfer between sites triggers removal of old-site access on the effective date of transfer.

4.7 **Periodic review.** Department heads must recertify all user access in their area every quarter (cut-off dates: 31 March, 30 June, 30 September, 31 December). Uncertified access is suspended on the 15th business day after cut-off.

4.8 **Secondary use.** Any extraction of identifiable patient data for purposes other than care, billing, or legally required reporting requires a Data Use Request approved by the Data Protection Officer. Approvals are logged and are valid for no more than 12 months.

4.9 **Monitoring.** MeridianChart access logs are reviewed by the Privacy Office using the following automated flags:
(a) access to a record of a colleague, relative, or any person sharing the user's surname and postcode;
(b) more than 40 distinct chart opens by a T1 user in one shift;
(c) any access to a record flagged "Staff-as-patient" without an active care relationship;
(d) any access to a record flagged "High-profile" without a documented care relationship.
Flagged events must be triaged within 3 business days.

4.10 **Training.** All workforce members must complete privacy and security training before first access and then every 12 months. Completion below 98% at any site on the last day of a quarter must be escalated to the Chief Operating Officer.

4.11 **Devices and media.**
(a) Patient data must not be stored on personal devices or personal email accounts.
(b) Removable media is disabled by default. Exceptions are issued by Information Security for no more than 14 days and must use hardware-encrypted drives.
(c) Printing of full charts requires a stated purpose in MeridianChart. Printed material containing patient data must be shredded or placed in locked disposal bins by the end of the shift.
(d) Messages containing patient data may be sent only through the MeridianChart secure messaging, the patient portal, or the approved encrypted email gateway.

---

## 5. DISCLOSURE TO THIRD PARTIES

5.1 Disclosure to anyone outside Meridian is permitted only where one of the following applies: treatment, payment, or operations disclosure covered by consent 3.2(a) or 3.2(b); disclosure required by law and approved by the Privacy Office; or disclosure authorised in writing by the patient.

5.2 Every processor must have a signed data processing or business associate agreement on file with the Privacy Office before receiving any patient data. Each agreement must require the processor to notify Meridian of a suspected breach within 24 hours (see 8.3).

5.3 Requests from police, courts, insurers, employers, or media must be forwarded to the Privacy Office on the same business day and must not be answered by site staff.

---

## 6. RETENTION AND DISPOSAL

6.1 Records are retained for the periods in the table below. Periods run from the date shown in the "Trigger" column.

| Ref | Record type | Retention period | Trigger |
|---|---|---|---|
| 6.1.1 | Adult clinical record (general) | 10 years | Last encounter |
| 6.1.2 | Paediatric clinical record | Until age 25, or 10 years from last encounter, whichever is later | Patient's date of birth / last encounter |
| 6.1.3 | Oncology clinical record | 15 years | Last encounter |
| 6.1.4 | Occupational health surveillance record | 30 years | End of employment relationship with the employer client |
| 6.1.5 | Maternity and obstetric record | 25 years | Date of delivery or end of pregnancy |
| 6.1.6 | Mental health and substance use treatment notes | 12 years | Last encounter |
| 6.1.7 | Diagnostic imaging (images and reports) | 10 years | Date of study |
| 6.1.8 | Laboratory results (final reports) | 10 years | Date of report |
| 6.1.9 | Pathology slides and tissue blocks | 5 years (slides), 10 years (blocks) | Date of report |
| 6.1.10 | Prescription and medication administration records | 8 years | Date of prescription |
| 6.1.11 | Billing, claims, and payment records | 8 years | End of the financial year of the transaction |
| 6.1.12 | Consent records (ConsentVault) | Life of the related clinical record plus 3 years | Date the clinical record is destroyed |
| 6.1.13 | Marketing consent and suppression lists | 3 years | Withdrawal or last confirmation |
| 6.1.14 | Patient access request files | 6 years | Closure of the request |
| 6.1.15 | Breach incident files | 6 years | Closure of the incident |
| 6.1.16 | MeridianChart access and audit logs | 6 years | Date of log entry |
| 6.1.17 | Telehealth session recordings (where consented) | 90 days, unless a clinician has filed a section of the recording to the chart | Date of session |
| 6.1.18 | Call-centre audio recordings | 90 days | Date of call |
| 6.1.19 | CCTV in clinical and reception areas | 30 days | Date of recording |
| 6.1.20 | Patient portal messages | Same as the clinical record | Last encounter |
| 6.1.21 | Appointment no-show and cancellation logs | 3 years | Date of appointment |
| 6.1.22 | De-identified research and quality datasets | 10 years | Date of dataset creation |
| 6.1.23 | Job applicant and unsuccessful-candidate records (not patient data) | 12 months | Date of decision |

6.2 **Litigation hold.** The General Counsel's office may place a hold on any record. A held record must not be destroyed regardless of its scheduled date. Holds are recorded in the Retention Register and reviewed every 6 months.

6.3 **Deceased patients.** The retention periods in 6.1 continue to apply after death. Requests from next of kin are handled under Section 7 with proof of relationship and a death certificate.

6.4 **Disposal schedule.**
(a) The Records Management Unit (RMU) runs the MeridianChart retention report on the first business day of each month and lists records that reached their destruction date in the preceding month.
(b) The department head and the Privacy Office must approve the destruction list within 15 business days.
(c) Approved records are destroyed within 30 days of approval.
(d) A Certificate of Destruction is signed by the RMU supervisor and retained for 10 years.

6.5 **Methods.**
(a) Paper: cross-cut shredding to particle size not exceeding 4 mm × 40 mm, or pulping through an approved contractor.
(b) Electronic: cryptographic erasure or overwriting per the Information Security Media Standard. Backup copies must expire within 35 days of production deletion.
(c) Physical media (drives, tapes): physical destruction with a certificate from the contractor, Brantford Secure Shred Ltd.

6.6 **Extended retention and early deletion.**
(a) Retention beyond a stated period is permitted only with written approval of the Data Protection Officer and a recorded reason.
(b) A patient request for deletion of a clinical record before the stated period will be declined in writing, citing the applicable clause of this Section, except for data held beyond what Section 6.1 requires (for example, marketing profiles after withdrawal), which must be deleted within 30 days of the request.

---

## 7. PATIENT DATA ACCESS REQUESTS

7.1 **Who may request.** A patient, a legally authorised representative, a parent or guardian of a minor (subject to 3.5(d)), or a person with written patient authorisation may request a copy of the patient's data held by Meridian.

7.2 **How requests are made.** Requests may be submitted through the patient portal, by the Privacy Office email address, by post to the Privacy Office, or in person at any site on Form DH-07. Staff must not refuse to accept a request because it is not on the form. A request made in any other manner must be forwarded to the Privacy Office on the same business day.

7.3 **Clock start.** The response period starts on the day the request is first received by any Meridian workforce member, not the day it reaches the Privacy Office.

7.4 **Logging and acknowledgement.**
(a) The Privacy Office logs every request in PrivacyDesk within 1 business day of receipt.
(b) A written acknowledgement with the reference number and the due date must be sent within 2 business days of receipt.

7.5 **Identity verification.**
(a) The requester must be verified against two identifiers from the registered record (such as date of birth, registered phone number, last visit date, or postcode), or by government photo ID where the request is made in person or by post.
(b) A verification query must be sent within 3 business days of receipt. The clock is paused only while Meridian awaits the requester's reply, to a maximum pause of 14 days. If there is no reply, the request is closed and the requester is told how to re-submit.
(c) Authorised representatives must provide documentary proof of authority.

7.6 **Response deadline.**
(a) Meridian must provide the response within **21 calendar days** of the clock start under 7.3.
(b) Where the request is complex (more than 5 years of records across more than 3 sites, or involving archived paper records), the DPO may approve a single extension of up to **9 additional calendar days**, giving a total of 30 days. The requester must be notified in writing, with reasons, no later than **day 14**.
(c) No further extensions are permitted. Requests that reach day 18 without a draft response must be escalated to the DPO, and at day 20 to the Chief Operating Officer.

7.7 **Retrieval.**
(a) Site managers must return records from their site within 5 business days of the Privacy Office's retrieval request.
(b) Processors must return records within 5 business days of Meridian's request.
(c) The RMU retrieves offsite paper records within 7 business days.

7.8 **Clinical review.** A treating clinician may review the record before release only to identify content that must be withheld under 7.10. Review must be completed within 5 business days of the request being sent to them. Clinician review is not a condition for release of records not subject to 7.10.

7.9 **Scope and format.**
(a) The response includes the clinical record, results, imaging reports, correspondence, billing statements, consent records, and the list of third parties to whom data has been disclosed in the previous 24 months.
(b) The format is that requested by the patient where reasonably producible, otherwise a password-protected PDF through the portal. Imaging is provided on encrypted media or through a secure download link valid for 14 days.
(c) Access-log data about which workforce members viewed a record is provided on request, limited to the previous 12 months, and limited to role and date rather than staff names unless a breach investigation has confirmed inappropriate access.

7.11 **Fees.**
(a) There is no charge for electronic copies.
(b) Paper copies are charged at INR 5 per page (S
