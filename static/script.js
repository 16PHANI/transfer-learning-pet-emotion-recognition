document.addEventListener('DOMContentLoaded', () => {
    // ---------------- Dark/Light Mode Toggle ----------------
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;

    // Load saved theme from localStorage
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        body.classList.remove('light-mode', 'dark-mode');
        body.classList.add(savedTheme);
        themeToggle.textContent = savedTheme === 'light-mode' ? '🌙' : '☀️';
    }

    themeToggle.addEventListener('click', () => {
        if (body.classList.contains('light-mode')) {
            body.classList.replace('light-mode', 'dark-mode');
            themeToggle.textContent = '☀️';
            localStorage.setItem('theme', 'dark-mode');
        } else {
            body.classList.replace('dark-mode', 'light-mode');
            themeToggle.textContent = '🌙';
            localStorage.setItem('theme', 'light-mode');
        }
    });

    // ---------------- Button Hover Animation ----------------
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(btn => {
        btn.addEventListener('mouseenter', () => {
            btn.style.transform = 'scale(1.05)';
        });
        btn.addEventListener('mouseleave', () => {
            btn.style.transform = 'scale(1)';
        });
    });

    // ---------------- Hero Image Hover Effect ----------------
    const heroImages = document.querySelectorAll('.hero-images img');
    heroImages.forEach(img => {
        img.addEventListener('mouseenter', () => {
            img.style.transform = 'scale(1.05)';
        });
        img.addEventListener('mouseleave', () => {
            img.style.transform = 'scale(1)';
        });
    });

    // ---------------- Daily Routine Card Toggle ----------------
    const routineCards = document.querySelectorAll('.routine-cards .card');
    routineCards.forEach(card => {
        card.addEventListener('click', () => {
            const content = card.querySelector('.card-content');
            if (content.style.display === 'block') {
                content.style.display = 'none';
            } else {
                content.style.display = 'block';
            }
        });
    });
});

