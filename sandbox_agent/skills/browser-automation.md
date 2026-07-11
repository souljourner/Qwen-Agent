# Browser Automation Guide
> Driving the in-container browser: navigate/screenshot/click/type/scroll workflow, login, 2FA/CAPTCHA handling.

For pages requiring JavaScript rendering (SPAs, dynamic content, login-required pages), use browser tools instead of web_url_fetch.

## Typical workflow for e-commerce / checkout
1. `browser_navigate(url="...")` — open the page
2. `browser_screenshot()` — take screenshot, examine what you see
3. `browser_click(text="Buy Now")` or `browser_click(x=400, y=520)` — interact with elements
4. `browser_screenshot()` — verify result after each action
5. `browser_type(text="...", into="Credit card")` — fill in forms
6. Repeat until task complete

## Tips
- Always screenshot after navigating, before clicking — the vision model needs to see the page layout
- Use `text=` for labeled buttons ("Add to Cart", "Checkout") — more reliable than coordinates
- Use coordinates `(x, y)` when you need precision on visual elements (icons, images)
- Scroll with `browser_scroll` if content is below the fold
- Cookies persist between runs: if you log in once, subsequent runs stay logged in
- For bulk scraping of many URLs, prefer code_interpreter with requests (faster). Browser is for interaction.
- Credentials: `browser_save_credentials` / `browser_get_credentials` store and recall site logins.

## 2FA and CAPTCHA handling
The agent cannot solve 2FA challenges or CAPTCHAs. When you encounter one:
1. Take a screenshot — identify the 2FA/CAPTCHA page visually
2. Call `request_user` to ask the user for the 2FA code or CAPTCHA solution
3. Once the user replies, proceed with the code/answer using `browser_type` and `browser_click`
4. After successful login, cookies persist — future runs won't need 2FA again (until cookies expire)
