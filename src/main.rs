use llm::{
    InferenceFeedback, InferenceParameters, InferenceRequest, InferenceResponse, Model, Prompt,
};

enum Models {
    WizardVicuna13BUncensoredQ41,
    WizardVicuna7BUncensoredQ40,
}

impl Models {
    fn get_path(&self) -> &'static str {
        match self {
            Models::WizardVicuna13BUncensoredQ41 => "models/Wizard-Vicuna-13B-Uncensored.ggmlv3.q4_1.bin", 
            Models::WizardVicuna7BUncensoredQ40 =>  "models/Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_0.bin",
        }
    }
}

fn main() {
    let llama = llm::load::<llm::models::Llama>(
        std::path::Path::new(Models::WizardVicuna13BUncensoredQ41.get_path()),
        Default::default(),
        None,
        llm::load_progress_callback_stdout,
    )
    .expect("Model should have loaded");

    let mut session = llama.start_session(Default::default());
    let result = session.infer::<std::convert::Infallible>(
        &llama,
        &mut rand::thread_rng(),
        &InferenceRequest {
            prompt: Prompt::Text(
                "USER: How can I create a variable in Rust that holds the first argument passed in when invoking the program?\nASSISTANT:",
            ),
            parameters: &mut InferenceParameters {
                temperature: 0.1,
                ..Default::default()
            },
            play_back_previous_tokens: false,
            maximum_token_count: None,
        },
        &mut Default::default(),
        |response| match response {
            InferenceResponse::EotToken => Ok(InferenceFeedback::Halt),
            InferenceResponse::SnapshotToken(s) => {
                print!("{s}");
                Ok(InferenceFeedback::Continue)
            }
            InferenceResponse::PromptToken(s) => {
                print!("{s}");
                Ok(InferenceFeedback::Continue)
            }
            InferenceResponse::InferredToken(s) => {
                print!("{s}");
                Ok(InferenceFeedback::Continue)
            }
        },
    );

    match result {
        Ok(result) => println!("Inference status: {:?}", result),
        Err(err) => println!("Inference error: {:?}", err),
    }
}
