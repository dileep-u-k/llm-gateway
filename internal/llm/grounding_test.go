package llm

import "testing"

func TestGroundingPolicyEngineAbstainsWhenGroundedAnswerLacksEvidence(t *testing.T) {
	engine := NewGroundingPolicyEngine(0.45)

	decision := engine.Decide(string(AnswerModeGroundedRequired), ExecutionIntent{GroundingRequired: true}, nil, nil)
	if !decision.ShouldAbstain {
		t.Fatal("expected grounded-required mode to abstain when no evidence is available")
	}
	if decision.EvidenceStatus != EvidenceStatusInsufficient {
		t.Fatalf("expected insufficient evidence, got %s", decision.EvidenceStatus)
	}
}

func TestGroundingPolicyEngineAcceptsSuccessfulToolOutput(t *testing.T) {
	engine := NewGroundingPolicyEngine(0.45)

	decision := engine.Decide(string(AnswerModeToolOutputPriority), ExecutionIntent{}, nil, []ContextToolResult{
		{Name: "getCurrentWeather", Status: "success", Summary: "Sunny 29C"},
	})
	if decision.ShouldAbstain {
		t.Fatal("expected successful tool output to satisfy tool-output-priority mode")
	}
	if decision.EvidenceStatus != EvidenceStatusSufficient {
		t.Fatalf("expected sufficient evidence, got %s", decision.EvidenceStatus)
	}
}

func TestFormatGroundedResponseAppendsSources(t *testing.T) {
	formatted := FormatGroundedResponse("Answer", &RetrievalResult{
		Provenance: []RetrievalProvenance{
			{DocTitle: "Architecture Notes", Source: "docs/arch.md", Section: "Routing"},
		},
	}, nil, GroundingDecision{Mode: AnswerModeGroundedPreferred})

	if formatted == "Answer" {
		t.Fatal("expected grounded response formatter to append sources")
	}
}
