[![Python application testing](https://github.com/yoyosasa512/BTC-PREDICTION-AND-VALIDATION/actions/workflows/python-app.yml/badge.svg)](https://github.com/yoyosasa512/BTC-PREDICTION-AND-VALIDATION/actions/workflows/python-app.yml)
### システム構成図

```mermaid
graph LR
    subgraph Local_Development ["ローカル開発環境"]
        A[VSCode / Python] -->|指示| B(Antigravity / AI)
        B -->|自動処理| C{Git Commit / Push}
    end

    subgraph GitHub_Platform ["GitHub (CI)"]
        C --> D[GitHub Repository]
        D -->|Trigger| E{GitHub Actions}
        subgraph CI_Process ["CIプロセス"]
            E --> F[Environment Setup]
            F --> G[Run pytest]
            G -->|Success| H[Build Status Badge: Passing]
        end
    end

    subgraph Hosting_Cloud ["Cloud (CD)"]
        H -->|Auto Deploy| I[Streamlit Cloud]
        I --> J((Public Web App))
    end

    style B fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style I fill:#dfd,stroke:#333,stroke-width:2px
