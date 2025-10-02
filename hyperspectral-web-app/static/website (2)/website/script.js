// Initialize Lucide icons
lucide.createIcons();

// Mobile menu functionality
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const mobileMenu = document.getElementById('mobileMenu');

mobileMenuBtn.addEventListener('click', () => {
    mobileMenu.classList.toggle('show');
});

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            const headerOffset = 80;
            const elementPosition = target.getBoundingClientRect().top;
            const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

            window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
            });
        }
        
        // Close mobile menu if open
        mobileMenu.classList.remove('show');
    });
});

// Header background on scroll
window.addEventListener('scroll', () => {
    const header = document.querySelector('.header');
    if (window.scrollY > 50) {
        header.style.background = 'rgba(255, 255, 255, 0.98)';
    } else {
        header.style.background = 'rgba(255, 255, 255, 0.95)';
    }
});

// Button click handlers (you can customize these)
// document.querySelectorAll('.btn').forEach(button => {
//     if (button.textContent.includes('Request Demo') || 
//         button.textContent.includes('Book Demo')) {
//         button.addEventListener('click', () => {
//             alert('Demo request functionality would be implemented here!');
//         });
//     } else if (button.textContent.includes('Learn More')) {
//         button.addEventListener('click', () => {
//             document.querySelector('#technology').scrollIntoView({ 
//                 behavior: 'smooth',
//                 block: 'start'
//             });
//         });
//     } else if (button.textContent.includes('Download Info')) {
//         button.addEventListener('click', () => {
//             alert('Download functionality would be implemented here!');
//         });
//     } else if (button.textContent.includes('Contact Us')) {
//         button.addEventListener('click', () => {
//             alert('Contact form would be implemented here!');
//         });
//     } else if (button.textContent.includes('Start Your Plastic Detection Journey')) {
//         button.addEventListener('click', () => {
//             alert('Registration process would be implemented here!');
//         });
//     }
// });

// Intersection Observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Add animation styles and observe elements
document.addEventListener('DOMContentLoaded', () => {
    const animateElements = document.querySelectorAll('.feature-card, .application-card, .cta-card');
    
    animateElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});
