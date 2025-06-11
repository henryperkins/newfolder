/**
 * Global declaration for CSS modules so that TypeScript recognises
 * imports such as `import styles from './Component.module.css'`.
 *
 * This avoids "Cannot find module '*.module.css'" compiler errors
 * while still providing typed class names as `string`.
 */

declare module '*.module.css' {
  // Using `Record<string, string>` keeps the mapping flexible while
  // preserving autocomplete for known class names in many editors.
  const classes: Record<string, string>;
  export default classes;
}

// Vite also supports plain `.css` imports without the `.module` suffix in
// component code.  We declare those too to cover all style variations.
declare module '*.css' {
  const classes: Record<string, string>;
  export default classes;
}
