const DEBUG_REQUESTS = false;

class FS
{
    static PROJECT_ID = '68b709f30027f98a667b';
    static END_POINT = 'https://cloud.appwrite.io/v1';

    static DATABASE_ID = '68b70c8f002d2fe08d90';
    static BUCKET_ID = '68b70c6b0004ae91e454';
    static SHADERS_COLLECTION_ID = 'shaders';
    static USERS_COLLECTION_ID = 'users';
    static ASSETS_COLLECTION_ID = 'assets';
    static INTERACTIONS_COLLECTION_ID = 'interactions';
    static COLLECTIONS_COLLECTION_ID = 'collections';

    constructor()
    {
        this.client = new Appwrite.Client()
            .setEndpoint( FS.END_POINT )
            .setProject( FS.PROJECT_ID );

        this.account = new Appwrite.Account( this.client );
        this.databases = new Appwrite.Databases( this.client );
        this.storage = new Appwrite.Storage( this.client );

        this._assetsCache = new Map();
    }

    getUserId()
    {
        return this.user['$id'];
    }

    async detectAutoLogin()
    {
        try
        {
            this.user = await this.account.get();
            this.session = await this.account.getSession( 'current' );
            console.log( 'Autologged', this.user );
        }
        catch ( err )
        {
            // No session
            console.warn( 'No current session' );
        }
    }

    async createAccount( email, password, name, oncreate, onerror )
    {
        console.assert( email && password && name );
        if ( DEBUG_REQUESTS ) this.logDebug( 'Request: createAccount', email );
        await this.account.create( {
            userId: 'unique()',
            email,
            password,
            name
        } ).then( ( user ) => {
            if ( oncreate ) oncreate( user );
        } )
            .catch( ( error ) => {
                console.error( error );
                if ( onerror ) onerror( error?.message );
            } );
    }

    async login( email, password, onlogin, onerror )
    {
        if ( DEBUG_REQUESTS ) this.logDebug( 'Request: login', email );
        await this.account.createEmailPasswordSession( {
            email,
            password
        } ).then( async ( session ) => {
            this.user = await this.account.get();
            this.session = session;
            console.log( 'Login', this.user );
            if ( onlogin ) onlogin( this.user, this.session );
        } )
            .catch( ( error ) => {
                if ( onerror ) onerror( error?.message );
            } );
    }

    async logout()
    {
        if ( !this.user )
        {
            return;
        }
        if ( DEBUG_REQUESTS ) this.logDebug( 'Request: logout' );
        try
        {
            const current = await this.account.getSession( 'current' );
            await this.account.deleteSession( {
                sessionId: current['$id']
            } );

            this.user = null;

            console.log( 'Logged out!' );
        }
        catch ( err )
        {
            console.error( err );
        }
    }

    async listDocuments( collectionId, queries = [] )
    {
        const key = queries.join( '_' );
        if ( collectionId === FS.ASSETS_COLLECTION_ID && queries.length )
        {
            if ( this._assetsCache.has( key ) )
            {
                return this._assetsCache.get( key );
            }
        }
        if ( DEBUG_REQUESTS ) this.logDebug( 'Request: listDocuments', collectionId, queries );
        const r = await this.databases.listDocuments( {
            databaseId: FS.DATABASE_ID,
            collectionId,
            queries
        } );
        if ( DEBUG_REQUESTS ) this.logResponse( 'Response: listDocuments', r );
        if ( collectionId === FS.ASSETS_COLLECTION_ID )
        {
            this._assetsCache.set( key, r );
        }
        return r;
    }

    async getDocument( collectionId, documentId )
    {
        if ( collectionId === FS.ASSETS_COLLECTION_ID )
        {
            if ( this._assetsCache.has( documentId ) )
            {
                return this._assetsCache.get( documentId );
            }
        }
        if ( DEBUG_REQUESTS ) this.logDebug( 'Request: getDocument', collectionId, documentId );
        const r = await this.databases.getDocument( {
            databaseId: FS.DATABASE_ID,
            collectionId,
            documentId
        } );
        if ( DEBUG_REQUESTS ) this.logResponse( 'Response: getDocument', r );
        if ( collectionId === FS.ASSETS_COLLECTION_ID )
        {
            this._assetsCache.set( documentId, r );
        }
        return r;
    }

    async createDocument( collectionId, data )
    {
        if ( DEBUG_REQUESTS ) this.logDebug( 'Request: createDocument', data );
        const r = await this.databases.createDocument( {
            databaseId: FS.DATABASE_ID,
            collectionId,
            documentId: 'unique()',
            data
        } );
        if ( DEBUG_REQUESTS ) this.logResponse( 'Response: createDocument', r );
        return r;
    }

    async updateDocument( collectionId, documentId, data )
    {
        if ( DEBUG_REQUESTS ) this.logDebug( 'Request: updateDocument', collectionId, documentId, data );
        const r = await this.databases.updateDocument( {
            databaseId: FS.DATABASE_ID,
            collectionId,
            documentId,
            data
        } );
        if ( DEBUG_REQUESTS ) this.logResponse( 'Response: updateDocument', r );
        return r;
    }

    async deleteDocument( collectionId, documentId )
    {
        if ( DEBUG_REQUESTS ) this.logDebug( 'Request: deleteDocument', collectionId, documentId );
        const r = await this.databases.deleteDocument( {
            databaseId: FS.DATABASE_ID,
            collectionId,
            documentId
        } );
        if ( DEBUG_REQUESTS ) this.logResponse( 'Response: deleteDocument', r );
        return r;
    }

    async listFiles( queries = [] )
    {
        if ( DEBUG_REQUESTS ) this.logDebug( 'Request: listFiles', queries );
        const r = await this.storage.listFiles( {
            bucketId: FS.BUCKET_ID,
            queries
        } );
        if ( DEBUG_REQUESTS ) this.logResponse( 'Response: listFiles', r );
        return r;
    }

    async getFile( fileId )
    {
        if ( DEBUG_REQUESTS ) this.logDebug( 'Request: getFile', fileId );
        const r = await this.storage.getFile( {
            bucketId: FS.BUCKET_ID,
            fileId
        } );
        if ( DEBUG_REQUESTS ) this.logResponse( 'Response: getFile', r );
        return r;
    }

    async getFileUrl( fileId )
    {
        if ( DEBUG_REQUESTS ) this.logDebug( 'Request: getFileUrl', fileId );
        const r = await this.storage.getFileView( {
            bucketId: FS.BUCKET_ID,
            fileId
        } );
        if ( DEBUG_REQUESTS ) this.logResponse( 'Response: getFileUrl', r );
        return r;
    }

    async getFileContent( fileId )
    {
        const url = await this.getFileUrl( fileId );
        return await this.requestFile( url );
    }

    async createFile( file, fileId )
    {
        if ( DEBUG_REQUESTS ) this.logDebug( 'Request: createFile', file, fileId );
        const r = await this.storage.createFile( {
            bucketId: FS.BUCKET_ID,
            fileId: fileId ?? 'unique()',
            file
        } );
        if ( DEBUG_REQUESTS ) this.logResponse( 'Response: createFile', r );
        return r;
    }

    async deleteFile( fileId )
    {
        if ( DEBUG_REQUESTS ) this.logDebug( 'Request: deleteFile', fileId );
        const r = await this.storage.deleteFile( {
            bucketId: FS.BUCKET_ID,
            fileId
        } );
        if ( DEBUG_REQUESTS ) this.logResponse( 'Response: deleteFile', r );
        return r;
    }

    async getImagePreview( fileId, options )
    {
        if ( DEBUG_REQUESTS ) this.logDebug( 'Request: getImagePreview', fileId, options );
        const r = await this.storage.getFilePreview( {
            bucketId: FS.BUCKET_ID,
            fileId,
            ...options
        } );
        if ( DEBUG_REQUESTS ) this.logResponse( 'Response: getImagePreview', r );
        return r;
    }

    async requestFile( url, dataType, nocache )
    {
        return new Promise( ( resolve, reject ) => {
            dataType = dataType ?? 'arraybuffer';
            const mimeType = dataType === 'arraybuffer' ? 'application/octet-stream' : undefined;
            var xhr = new XMLHttpRequest();
            xhr.open( 'GET', url, true );
            xhr.responseType = dataType;
            if ( mimeType )
            {
                xhr.overrideMimeType( mimeType );
            }
            if ( nocache )
            {
                xhr.setRequestHeader( 'Cache-Control', 'no-cache' );
            }
            xhr.onload = function( load )
            {
                var response = this.response;
                if ( this.status != 200 )
                {
                    var err = 'Error ' + this.status;
                    reject( err );
                    return;
                }
                resolve( response );
            };
            xhr.onerror = function( err )
            {
                reject( err );
            };
            xhr.send();
            return xhr;
        } );
    }

    logDebug( title, ...data )
    {
        console.log(
            `%cAppwrite%c ${title}`,
            'background:#111827;color:#60a5fa;padding:2px 6px;border-radius:4px;font-weight:600',
            'color:#9ca3af;font-weight:500',
            ...data
        );
    }

    logResponse( title, r )
    {
        if ( r ) this.logSuccess( title, r );
        else this.logError( title, r );
    }

    logSuccess( title, ...data )
    {
        console.log(
            `%cAppwrite%c ${title}`,
            'background:#112718;color:#60faa5;padding:2px 6px;border-radius:4px;font-weight:600',
            'color:#34d399;font-weight:500',
            ...data
        );
    }

    logError( title, ...data )
    {
        console.log(
            `%cAppwrite%c ${title}`,
            'background:#271811;color:#60a5fa;padding:2px 6px;border-radius:4px;font-weight:600',
            'color:#ef4444;font-weight:500',
            ...data
        );
    }

    async _downloadDatabasesBackup()
    {
        const collections = [
            { id: FS.SHADERS_COLLECTION_ID, name: 'shaders' },
            { id: FS.USERS_COLLECTION_ID, name: 'users' },
            { id: FS.ASSETS_COLLECTION_ID, name: 'assets' },
            { id: FS.INTERACTIONS_COLLECTION_ID, name: 'interactions' },
            { id: FS.COLLECTIONS_COLLECTION_ID, name: 'collections' },
        ];

        const backup = {
            timestamp: new Date().toISOString(),
            database: FS.DATABASE_ID,
            collections: {}
        };

        for ( const col of collections )
        {
            console.log( `Backing up ${ col.name }...` );
            const docs = [];
            const limit = 100;
            let offset = 0;
            let total = Infinity;

            while ( offset < total )
            {
                const response = await this.databases.listDocuments( {
                    databaseId: FS.DATABASE_ID,
                    collectionId: col.id,
                    queries: [
                        Appwrite.Query.limit( limit ),
                        Appwrite.Query.offset( offset )
                    ]
                } );
                total = response.total;
                docs.push( ...response.documents );
                offset += limit;
            }

            backup.collections[ col.name ] = { total: docs.length, documents: docs };
            console.log( `  ${ col.name }: ${ docs.length } documents` );
        }

        const json = JSON.stringify( backup, null, 2 );
        const blob = new Blob( [ json ], { type: 'application/json' } );
        const url = URL.createObjectURL( blob );
        const a = document.createElement( 'a' );
        a.href = url;
        a.download = `shaderhub-dbs-${ new Date().toISOString().slice( 0, 10 ) }.json`;
        a.click();
        URL.revokeObjectURL( url );

        console.log( 'DB backup downloaded!' );
    }

    async _downloadFilesBackup()
    {
        const zip = new JSZip();
        const files = [];
        let fileOffset = 0;
        let fileTotal = Infinity;

        while ( fileOffset < fileTotal )
        {
            const response = await this.storage.listFiles( {
                bucketId: FS.BUCKET_ID,
                queries: [
                    Appwrite.Query.limit( 100 ),
                    Appwrite.Query.offset( fileOffset )
                ]
            } );
            fileTotal = response.total;
            files.push( ...response.files );
            fileOffset += 100;
        }

        console.log( `  Found ${ files.length } files, downloading...` );

        for ( let i = 0; i < files.length; i++ )
        {
            const file = files[ i ];
            console.log( `  [${ i + 1 }/${ files.length }] ${ file.name }` );
            try
            {
                const data = await this.storage.getFileDownload( {
                    bucketId: FS.BUCKET_ID,
                    fileId: file['$id']
                } );
                zip.file( file.name, data );
            }
            catch ( err )
            {
                console.error( `  Failed to download ${ file.name }:`, err.message );
            }
        }

        const zipBlob = await zip.generateAsync( { type: 'blob' } );
        const zipUrl = URL.createObjectURL( zipBlob );
        const zipLink = document.createElement( 'a' );
        zipLink.href = zipUrl;
        zipLink.download = `shaderhub-files-${ new Date().toISOString().slice( 0, 10 ) }.zip`;
        zipLink.click();
        URL.revokeObjectURL( zipUrl );

        console.log( 'Storage backup downloaded!' );
    }

    _downloadBackup()
    {
        if ( !this.user || !this.user['$id'] ) return alert( 'You must be logged in to download backups.' );
        this._downloadDatabasesBackup();
        this._downloadFilesBackup();
    }
}

export { FS };
