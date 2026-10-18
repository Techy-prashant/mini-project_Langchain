# 📊 Evaluation Benchmark & Verified Results

This document contains the verified evaluation benchmark results conducted against the prescribed test queries for the TechNova Solutions Employee Policy document.

---

## Benchmark Query 1

**Question:**  
> "How many paid annual leave days does a full-time employee receive per year?"

### Retrieved Context Chunks (Top 3 via FAISS)
- **[Chunk 1]:**  
  `1. Annual Leave: Every full-time employee receives 18 paid annual leave days per calendar year. Employees must submit annual leave requests to their manager at least 3 calendar days before the requested leave date. Leave approval depends on the manager's confirmation.`
- **[Chunk 2]:**  
  `4. Sick Leave: Employees receive 10 paid sick leave days per calendar year. If an employee is absent due to illness for more than two consecutive working days, a medical certificate must be submitted to HR.`
- **[Chunk 3]:**  
  `3. Working Hours: The standard working day is from 9:30 AM to 5:30 PM, Monday through Friday. Employees receive a one-hour lunch break.`

### Generated Grounded Response
> Based on TechNova Solutions' policy, every full-time employee receives **18 paid annual leave days** per calendar year.

---

## Benchmark Query 2

**Question:**  
> "An employee wants to work remotely on Wednesday and take leave on Friday. What do the company policies say about these two requests?"

### Retrieved Context Chunks (Top 3 via FAISS)
- **[Chunk 1]:**  
  `2. Work From Home: Employees may work remotely on Wednesdays and Fridays. Employees must inform their manager by email at least one working day in advance. Remote work is subject to manager approval.`
- **[Chunk 2]:**  
  `1. Annual Leave: Every full-time employee receives 18 paid annual leave days per calendar year. Employees must submit annual leave requests to their manager at least 3 calendar days before the requested leave date. Leave approval depends on the manager's confirmation.`
- **[Chunk 3]:**  
  `4. Sick Leave: Employees receive 10 paid sick leave days per calendar year. If an employee is absent due to illness for more than two consecutive working days, a medical certificate must be submitted to HR.`

### Generated Grounded Response
> Based on TechNova Solutions' policy context, here is the information for each request:
> 
> ### 1. Working Remotely on Wednesday
> * **Eligibility:** Employees are permitted to work remotely on Wednesdays (and Fridays).
> * **Notice Period:** Employees must inform their manager at least one working day in advance.
> * **Submission Method:** Notice must be sent to the manager by email.
> * **Manager Approval:** Remote work is subject to manager approval and is not automatically approved.
> 
> ### 2. Taking Leave on Friday (Annual Leave)
> * **Notice Period:** Employees must submit their request at least 3 calendar days before the requested leave date.
> * **Submission Method:** The request must be submitted directly to the employee's manager.
> * **Manager Approval:** Leave approval depends on the manager's confirmation and is not automatically approved.

---

## Benchmark Query 3 (Absence Test / Anti-Hallucination)

**Question:**  
> "What is the company's parental leave policy?"

### Retrieved Context Chunks (Top 3 via FAISS)
- **[Chunk 1]:**  
  `1. Annual Leave: Every full-time employee receives 18 paid annual leave days per calendar year. Employees must submit annual leave requests to their manager at least 3 calendar days before the requested leave date. Leave approval depends on the manager's confirmation.`
- **[Chunk 2]:**  
  `4. Sick Leave: Employees receive 10 paid sick leave days per calendar year. If an employee is absent due to illness for more than two consecutive working days, a medical certificate must be submitted to HR.`
- **[Chunk 3]:**  
  `3. Working Hours: The standard working day is from 9:30 AM to 5:30 PM, Monday through Friday. Employees receive a one-hour lunch break.`

### Generated Grounded Response
> **TechNova Solutions' policy document does not provide information regarding the parental leave policy.**

---

## Verification Summary
- **Factual Accuracy:** 100%
- **Hallucination Rate:** 0.0%
- **Policy Absence Handling:** Passed (strictly refuses to invent policies)
