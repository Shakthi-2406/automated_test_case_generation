mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = \$PORT\n\
enableXsrfProtection=false\n\
enableCORS = false\n\
" > ~/.streamlit/config.toml