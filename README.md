# Assist, a Government Communications initiative

Assist is a tool built by government communicators, for government communicators. Powered by generative AI, Assist is a secure and accessible bespoke conversational tool that empowers users to become even more efficient and effective in their roles through helping to brainstorm, create first drafts and review work. By using Assist, GCS members also benefit from the confidence that its outputs follow GCS policies and standards. 

In addition to being able to reference specific GCS documents, responsible use has been the guiding principle of Assist’s development, informed by the GCS Framework for Ethical Innovation and GCS Generative AI Policy. Before using the tool, all GCS members must complete a bespoke ‘AI for Communicators’ training course, designed to upskill and inform them of the safe use of Assist and AI in the workplace. Assist provides users with more than 50 communications-specific ‘pre-built prompts’, which reflect the typical tasks a government communicator might need to do on a daily basis. These prompts span across all seven GCS disciplines, ensuring the tool is tailored to every government communicator use case. 

[Read the full blog here.](https://gcs.civilservice.gov.uk/blog/introducing-assist-the-dynamic-ai-tool-rapidly-transforming-government-communications/)

![image](https://github.com/user-attachments/assets/04e93ecc-d537-47a0-975f-7c779e54b6f5)

## Assist API quickstart
Requires `make`, `docker` and `docker-compose`.

- Run `make start` from the root directory. This will cause the Docker images to build and launch.
- Visit `localhost:5312/docs` from your browser.

The API is designed to work with the Connect frontend (another service created by Government Communications) though it can be used with other clients.
