# @AppleSupport Intent Taxonomy Specification

## 1. Overview & Operational Objectives
This taxonomy defines 8 operationally mutually-exclusive and collectively-exhaustive customer support intents discovered from empirical analysis of 103,842 reconstructed customer interactions with `@AppleSupport`.

The taxonomy balances:
* **Operational Granularity**: High enough to direct historical retrieval to relevant troubleshooting steps.
* **Separability**: Clear semantic boundaries preventing confusion between software glitches and hardware damage.
* **Actionability**: Each intent corresponds to concrete diagnostics, documentation links, or escalation workflows.

---

## 2. Intent Specifications

### 1. `SOFTWARE_UPDATE_OS`
* **Definition**: Inquiries regarding operating system updates (iOS, macOS, watchOS, tvOS), installation hangs, error codes during update verification, or features malfunctioning following an update.
* **Inclusion Criteria**:
  * Explicit mentions of OS versions (e.g., "iOS 11.2", "High Sierra", "beta").
  * System alerts such as "Unable to Verify Update" or "Update Paused".
  * Direct attribution of newly appeared glitches to a recent system update.
* **Exclusion Criteria**:
  * General app crashes from third-party App Store apps (use `DEVICE_PERFORMANCE_CRASH`).
  * Battery drain occurring generally without update attribution (use `BATTERY_POWER_CHARGING`).
* **Default Action**: AUTO_HANDLE with verified Apple Knowledge Base steps (restart, check storage, iTunes recovery update).

---

### 2. `BATTERY_POWER_CHARGING`
* **Definition**: Inquiries regarding rapid power depletion, battery capacity degradation, device refusing to charge, or overheating while charging or in standby.
* **Inclusion Criteria**:
  * Battery percentage dropping dramatically or device powering down at 20–30%.
  * "Accessory Not Supported" alerts when connecting lightning cable.
  * Overheating warnings on the display.
* **Exclusion Criteria**:
  * Physical port broken or foreign object lodged in port (use `HARDWARE_PHYSICAL_DAMAGE`).
* **Default Action**: AUTO_HANDLE with battery health diagnostics; ESCALATE if severe swelling or safety risk is reported.

---

### 3. `APPLE_ID_ACCOUNT_SECURITY`
* **Definition**: Issues involving Apple ID login credentials, two-factor authentication (2FA), account lockouts, verification codes, and iCloud storage or sync failures.
* **Inclusion Criteria**:
  * "Apple ID has been locked for security reasons".
  * Not receiving 2FA SMS or notification prompts.
  * Forgotten password or account recovery process.
  * iCloud backup failing or photos not syncing.
* **Exclusion Criteria**:
  * App Store billing disputes without account lockout (use `APP_STORE_BILLING_SUBSCRIPTIONS`).
* **Default Action**: Provide official iforgot.apple.com link; ESCALATE if the customer reports suspicious unauthorized account takeover.

---

### 4. `AUDIO_CONNECTIVITY_BLUETOOTH`
* **Definition**: Sound hardware/software issues, AirPods pairing, Bluetooth accessories, Wi-Fi connectivity, or cellular reception drops.
* **Inclusion Criteria**:
  * AirPods not connecting, cutting out, or one pod silent.
  * Receiver or front speaker crackling, distortion, or low call volume.
  * Wi-Fi greyed out or repeatedly disconnecting.
  * "No Service" or "Searching..." carrier connection issues.
* **Exclusion Criteria**:
  * Smashed display with broken speaker mesh (use `HARDWARE_PHYSICAL_DAMAGE`).
* **Default Action**: AUTO_HANDLE with network/audio reset and Bluetooth forget/re-pair troubleshooting procedures.

---

### 5. `HARDWARE_PHYSICAL_DAMAGE`
* **Definition**: Physical damage to device chassis, broken display glass, liquid damage, non-functional physical buttons (home/volume), or Genius Bar hardware repair requests.
* **Inclusion Criteria**:
  * Shattered, cracked, or bleeding OLED/LCD screen.
  * Submersion in water or liquid contact indicators triggered.
  * Physically jammed or stuck buttons.
  * Pricing and booking questions for physical Apple Store repairs.
* **Exclusion Criteria**:
  * Display unresponsive due to software freeze without physical impact (use `DEVICE_PERFORMANCE_CRASH`).
* **Default Action**: ESCALATE TO HUMAN / GENIUS BAR APPOINTMENT (AI cannot physically inspect or repair hardware).

---

### 6. `APP_STORE_BILLING_SUBSCRIPTIONS`
* **Definition**: Inquiries regarding iTunes / App Store monetary charges, unexpected recurring subscriptions, refund claims, Apple Pay transaction errors, or in-app purchase delivery issues.
* **Inclusion Criteria**:
  * Unrecognized credit card charge from "itunes.com/bill".
  * Requests to cancel subscription or demand a refund for accidental child purchase.
  * Apple Pay card verification failures.
* **Exclusion Criteria**:
  * General Apple ID login issues without monetary transactions (use `APPLE_ID_ACCOUNT_SECURITY`).
* **Default Action**: Provide reportaproblem.apple.com refund flow; ESCALATE if financial dispute or unauthorized debit is contested.

---

### 7. `DEVICE_PERFORMANCE_CRASH`
* **Definition**: Operating instability including boot loops (stuck on Apple logo), total screen unresponsiveness (frozen UI), black screen of death, or apps immediately terminating on launch.
* **Inclusion Criteria**:
  * Phone rebooting continuously on the white Apple logo.
  * Touchscreen completely unresponsive to touch or swipe gestures.
  * Native apps (Camera, Messages) crashing instantly upon opening.
* **Exclusion Criteria**:
  * Issues specifically occurring during an active iOS installation (use `SOFTWARE_UPDATE_OS`).
* **Default Action**: AUTO_HANDLE with force restart combinations (e.g. Volume Up, Volume Down, Hold Power) and recovery mode DFU steps.

---

### 8. `OTHER_OR_UNCLEAR`
* **Definition**: Ambiguous inquiries, emotional rants without actionable symptoms, general brand praise or criticism, or queries unrelated to technical device support.
* **Inclusion Criteria**:
  * "Apple please help me" (no symptom given).
  * Vague complaints ("Worst phone ever made").
  * Marketing questions ("When will the iPhone red edition launch?").
* **Exclusion Criteria**:
  * Any inquiry containing symptoms corresponding to intents 1–7.
* **Default Action**: Prompt customer for specific device model and iOS version, or ESCALATE if highly agitated.
