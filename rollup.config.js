import terser from '@rollup/plugin-terser';

export default [
    {
        input: 'src/app.js',
        external: [
            'lexgui',
            'lexgui/extensions/CodeEditor.js'
        ],
        output: {
            file: 'build/app.js',
            format: 'esm',
            sourcemap: false,
            plugins: [terser()]
        }
    }
];
