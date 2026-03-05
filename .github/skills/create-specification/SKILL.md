---
name: create-specification
description: 'As a product manager, create a human readable specification file for the solution, optimized for Generative AI consumption.'
---

# Create Specification

Your goal is to create a new specification file for `${input:SpecPurpose}`.

The specification file must define the requirements, constraints, and interfaces for the solution components in a manner that is clear, unambiguous, and structured for effective use by Generative AIs. 

The content should be focused on **what** needs to be achieved rather than **how** it should be implemented, allowing for flexibility in design and implementation while ensuring that all necessary information is provided for AI-driven development and decision-making.

Act as a product manager and as a software architect. Focus on defining the problem, requirements, constraints, and interfaces clearly and concisely. Describe the solution at a detail level that allows for effective implementation, testing and validation by development teams and Generative AIs. 

Describe the technologies used at a high level. 

Describe the screens and user interactions at a high level, without going into specific UI design details.

Describe UX principles and guidelines to be followed, without dictating specific design patterns or components.

## Best Practices for AI-Ready Specifications

- Use precise, explicit, and unambiguous language.
- Include examples and edge cases where applicable.
- Ensure the document is self-contained and does not rely on external context.

The specification should be saved in the [/spec/](/spec/) directory and named according to the following convention: `spec-[a-z0-9-]+.md`, where the name should be descriptive of the specification's content and starting with the highlevel purpose, which is one of [schema, tool, data, infrastructure, process, architecture, or design].

The specification file must be formatted in well formed Markdown.

Specification files must follow the template below, ensuring that all sections are filled out appropriately. The front matter for the markdown should be structured correctly as per the example following:

```md
---
title: [Concise Title Describing the Specification's Focus]
version: [Optional: e.g., 1.0, Date]
date_created: [YYYY-MM-DD]
last_updated: [Optional: YYYY-MM-DD]
owner: [Optional: Team/Individual responsible for this spec]
tags: [Optional: List of relevant tags or categories, e.g., `infrastructure`, `process`, `design`, `app` etc]
---

# Introduction

[A short concise introduction to the specification and the goal it is intended to achieve.]

## 1. Purpose & Scope

Provide a clear, concise description of the specification's purpose and the scope of its application. State the intended audience and any assumptions.

## 2. Definitions

List and define all acronyms, abbreviations, and domain-specific terms used in this specification.

## 3. Requirements, Constraints & Guidelines

Explicitly list all requirements, constraints, rules, and guidelines.

## 4. Interfaces & Data Contracts

Describe the interfaces, APIs, data contracts, or integration points. Use tables or code blocks for schemas and examples.

## 5. Acceptance Criteria

Define clear, testable acceptance criteria for each requirement using Given-When-Then format where appropriate.

## 6. Validation Criteria

[List the criteria or tests that must be satisfied for compliance with this specification.]

## 7. Related Specifications / Further Reading

[Link to related spec 1]
[Link to relevant external documentation]

```